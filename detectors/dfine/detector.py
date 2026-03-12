"""DFINEDetector: D-FINE object detection wrapper for BoT-SORT.

Supports three inference backends:
  - "onnx"    : ONNX Runtime
  - "trt"     : TensorRT engine
  - "pytorch" : PyTorch checkpoint via bundled TCB YAMLConfig

All backends expose a common ``detect(img_bgr)`` interface returning an
(N, 6) numpy array ``[x1, y1, x2, y2, score, class_id]`` in
original-image pixel coordinates.
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from detectors.base import BaseDetector

logger = logging.getLogger(__name__)

_IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_IMG_SCALE = 255.0

_LIB_DIR = Path(__file__).resolve().parent / "_lib"


# ---------------------------------------------------------------------------
# Standalone preprocessing
# ---------------------------------------------------------------------------

def _letterbox(
    img_rgb: np.ndarray,
    target_size: Tuple[int, int],
    fill: int = 0,
) -> Tuple[np.ndarray, Dict]:
    """Aspect-ratio-preserving resize with bottom-right padding."""
    import cv2

    target_h, target_w = target_size
    orig_h, orig_w = img_rgb.shape[:2]

    scale = min(target_h / orig_h, target_w / orig_w)
    resized_h = max(1, int(math.floor(orig_h * scale)))
    resized_w = max(1, int(math.floor(orig_w * scale)))

    resized = cv2.resize(img_rgb, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((target_h, target_w, 3), fill, dtype=np.uint8)
    canvas[:resized_h, :resized_w] = resized

    info = {
        "orig_size": (orig_h, orig_w),
        "resized_size": (resized_h, resized_w),
        "padding": [0, 0, target_w - resized_w, target_h - resized_h],
        "scale": scale,
    }
    return canvas, info


def _normalize_to_nchw(img_rgb: np.ndarray) -> np.ndarray:
    """Scale 0-255 -> ImageNet-normalized NCHW float32."""
    img = img_rgb.astype(np.float32) / _IMG_SCALE
    img = (img - _IMG_MEAN) / _IMG_STD
    img = img.transpose(2, 0, 1)[np.newaxis]
    return np.ascontiguousarray(img, dtype=np.float32)


# ---------------------------------------------------------------------------
# Standalone postprocessing
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))


def _decode_raw_outputs(
    pred_logits: np.ndarray,
    pred_boxes: np.ndarray,
    target_size: Tuple[int, int],
    num_top: int = 300,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert raw D-FINE outputs to scored xyxy detections."""
    if pred_logits.ndim == 3:
        pred_logits = pred_logits[0]
    if pred_boxes.ndim == 3:
        pred_boxes = pred_boxes[0]

    scores_all = _sigmoid(pred_logits)
    flat_scores = scores_all.flatten()

    num_top_actual = min(num_top, len(flat_scores))
    top_idx = np.argpartition(flat_scores, -num_top_actual)[-num_top_actual:]
    top_idx = top_idx[np.argsort(flat_scores[top_idx])[::-1]]

    n_classes = scores_all.shape[1]
    query_indices = top_idx // n_classes
    class_indices = top_idx % n_classes

    scores = flat_scores[top_idx].astype(np.float32)
    labels = class_indices.astype(np.int32)

    boxes_cxcywh = pred_boxes[query_indices]
    h, w = target_size
    cx = boxes_cxcywh[:, 0] * w
    cy = boxes_cxcywh[:, 1] * h
    bw = boxes_cxcywh[:, 2] * w
    bh = boxes_cxcywh[:, 3] * h

    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2

    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
    return labels, boxes_xyxy, scores


def _undo_letterbox(boxes: np.ndarray, letterbox_info: Dict) -> np.ndarray:
    """Map xyxy boxes from letterbox-canvas space to original-image space."""
    if boxes.size == 0:
        return boxes

    boxes = boxes.copy()
    pad_left, pad_top = letterbox_info["padding"][0], letterbox_info["padding"][1]
    scale = letterbox_info["scale"]
    orig_h, orig_w = letterbox_info["orig_size"]

    boxes[:, 0] -= pad_left
    boxes[:, 2] -= pad_left
    boxes[:, 1] -= pad_top
    boxes[:, 3] -= pad_top

    boxes /= scale

    boxes[:, 0] = np.clip(boxes[:, 0], 0.0, orig_w)
    boxes[:, 2] = np.clip(boxes[:, 2], 0.0, orig_w)
    boxes[:, 1] = np.clip(boxes[:, 1], 0.0, orig_h)
    boxes[:, 3] = np.clip(boxes[:, 3], 0.0, orig_h)

    return boxes


# ---------------------------------------------------------------------------
# Main wrapper
# ---------------------------------------------------------------------------

class DFINEDetector(BaseDetector):
    """D-FINE detection wrapper conforming to :class:`BaseDetector`."""

    BACKENDS: List[str] = ["onnx", "trt", "pytorch"]

    def __init__(
        self,
        backend: str = "onnx",
        model_path: Optional[str] = None,
        config_path: Optional[str] = None,
        target_size: Tuple[int, int] = (720, 1280),
        num_classes: int = 5,
        conf_threshold: float = 0.3,
        num_top_queries: int = 300,
        device: str = "cuda:0",
        fp16: bool = False,
        use_ema: bool = True,
    ) -> None:
        if backend not in self.BACKENDS:
            raise ValueError(f"backend must be one of {self.BACKENDS}, got '{backend}'")

        self.backend = backend
        self.model_path = str(model_path) if model_path else None
        self.config_path = str(config_path) if config_path else None
        self.target_size = target_size
        self.num_classes = num_classes
        self.conf_threshold = conf_threshold
        self.num_top_queries = num_top_queries
        self.device_str = device
        self.fp16 = fp16
        self.use_ema = use_ema

        self._model = None
        self._ort_session = None
        self._trt_ctx = None

        self._load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        if self.backend == "onnx":
            self._load_onnx()
        elif self.backend == "trt":
            self._load_trt()
        elif self.backend == "pytorch":
            self._load_pytorch()

    def _load_onnx(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "onnxruntime is required for the ONNX backend. "
                "Install with: pip install onnxruntime-gpu"
            ) from e

        if not self.model_path:
            raise ValueError("model_path must be provided for the ONNX backend.")

        providers: List[str] = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "cuda" in self.device_str.lower()
            else ["CPUExecutionProvider"]
        )
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._ort_session = ort.InferenceSession(
            self.model_path, sess_options=sess_opts, providers=providers
        )
        self._onnx_input_name = self._ort_session.get_inputs()[0].name
        logger.info("D-FINE ONNX session loaded from %s", self.model_path)

    def _load_trt(self) -> None:
        try:
            import tensorrt as trt
            import pycuda.autoinit  # noqa: F401
            import pycuda.driver as cuda
        except ImportError as e:
            raise ImportError(
                "tensorrt and pycuda are required for the TRT backend. "
                "Install pycuda with: pip install pycuda"
            ) from e

        if not self.model_path:
            raise ValueError("model_path must be provided for the TRT backend.")

        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)

        with open(self.model_path, "rb") as f:
            engine_data = f.read()

        engine = runtime.deserialize_cuda_engine(engine_data)
        context = engine.create_execution_context()

        host_inputs: Dict[str, np.ndarray] = {}
        host_outputs: Dict[str, np.ndarray] = {}
        dev_inputs: Dict[str, object] = {}
        dev_outputs: Dict[str, object] = {}
        tensor_shapes: Dict[str, tuple] = {}

        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            shape = tuple(engine.get_tensor_shape(name))
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            size = int(np.prod(shape)) * np.dtype(dtype).itemsize

            host_buf = cuda.pagelocked_empty(int(np.prod(shape)), dtype)
            dev_buf = cuda.mem_alloc(size)
            tensor_shapes[name] = shape

            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                host_inputs[name] = host_buf
                dev_inputs[name] = dev_buf
            else:
                host_outputs[name] = host_buf
                dev_outputs[name] = dev_buf

            context.set_tensor_address(name, int(dev_buf))

        self._trt_ctx = {
            "engine": engine,
            "context": context,
            "stream": cuda.Stream(),
            "host_inputs": host_inputs,
            "host_outputs": host_outputs,
            "dev_inputs": dev_inputs,
            "dev_outputs": dev_outputs,
            "tensor_shapes": tensor_shapes,
            "cuda": cuda,
        }
        logger.info("D-FINE TRT engine loaded from %s", self.model_path)

    def _load_pytorch(self) -> None:
        if not self.config_path:
            raise ValueError("config_path must be provided for the pytorch backend.")

        import importlib
        import os
        import torch.nn as nn

        lib_dir = str(_LIB_DIR)
        abs_config_path = str(Path(self.config_path).resolve())

        saved_cwd = os.getcwd()
        saved_path = sys.path.copy()

        _lib_top_level = ("models", "data", "core", "training", "utils")
        stashed_modules: Dict[str, object] = {}
        for prefix in _lib_top_level:
            for key in list(sys.modules):
                if key == prefix or key.startswith(prefix + "."):
                    stashed_modules[key] = sys.modules.pop(key)

        try:
            stdlib_paths = [
                p for p in saved_path
                if "site-packages" in p or "lib/python" in p or p == ""
            ]
            sys.path = [lib_dir] + stdlib_paths

            os.chdir(lib_dir)
            importlib.invalidate_caches()

            importlib.import_module("models.DFINE.nn")

            from core import YAMLConfig  # type: ignore[import]
        finally:
            os.chdir(saved_cwd)
            sys.path = saved_path

        update_dict = {"num_classes": self.num_classes}
        cfg = YAMLConfig(abs_config_path, **update_dict)
        for name in ("HGNetv2", "DFINEHGNetv2", "DomeHGNetv2"):
            if name in cfg.yaml_cfg and isinstance(cfg.yaml_cfg[name], dict):
                cfg.yaml_cfg[name]["pretrained"] = False

        if not self.model_path:
            raise ValueError("model_path (.pth) must be provided for the pytorch backend.")

        state_dict = self._load_pth_state(self.model_path)
        missing, unexpected = cfg.model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("Missing keys (%d): %s ...", len(missing), missing[:5])
        if unexpected:
            logger.warning("Unexpected keys (%d): %s ...", len(unexpected), unexpected[:5])

        device = torch.device(self.device_str)

        class _DeployWrapper(nn.Module):
            def __init__(self_, model_: nn.Module, postproc_: nn.Module) -> None:
                super().__init__()
                self_.model = model_.deploy() if hasattr(model_, "deploy") else model_.eval()
                self_.postproc = (
                    postproc_.deploy() if hasattr(postproc_, "deploy") else postproc_.eval()
                )

            def forward(self_, images: torch.Tensor, orig_target_sizes: torch.Tensor):
                outputs = self_.model(images)
                try:
                    res = self_.postproc(outputs, orig_target_sizes)
                except TypeError:
                    res = self_.postproc(outputs, orig_sizes=orig_target_sizes)
                if isinstance(res, tuple) and len(res) == 3:
                    return res
                if isinstance(res, dict):
                    if {"pred_labels", "pred_boxes", "pred_scores"} <= set(res):
                        return res["pred_labels"], res["pred_boxes"], res["pred_scores"]
                    if {"labels", "boxes", "scores"} <= set(res):
                        return (
                            res["labels"].unsqueeze(0),
                            res["boxes"].unsqueeze(0),
                            res["scores"].unsqueeze(0),
                        )
                raise TypeError(f"Unsupported postprocessor output: {type(res)}")

        model = _DeployWrapper(cfg.model.to(device), cfg.postprocessor.to(device))
        if self.fp16:
            model = model.half()
        model.eval()
        self._model = model
        logger.info("D-FINE PyTorch model loaded from %s", self.model_path)

    def _load_pth_state(self, pth_path: str) -> Dict[str, torch.Tensor]:
        ckpt = torch.load(pth_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            if (
                self.use_ema
                and "ema" in ckpt
                and isinstance(ckpt["ema"], dict)
                and "module" in ckpt["ema"]
            ):
                state = ckpt["ema"]["module"]
            elif "model" in ckpt and isinstance(ckpt["model"], dict):
                state = ckpt["model"]
            elif "state_dict" in ckpt:
                state = ckpt["state_dict"]
            else:
                state = {k: v for k, v in ckpt.items() if isinstance(v, torch.Tensor)}
        else:
            raise ValueError(f"Unsupported checkpoint type: {type(ckpt)}")

        return {
            (k[7:] if k.startswith("module.") else k): v
            for k, v in state.items()
        }

    # ------------------------------------------------------------------
    # Preprocessing / Postprocessing
    # ------------------------------------------------------------------

    def _preprocess(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, Dict]:
        import cv2

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        letterboxed, lb_info = _letterbox(img_rgb, self.target_size)
        tensor = _normalize_to_nchw(letterboxed)
        return tensor, lb_info

    def _postprocess(
        self,
        pred_logits: np.ndarray,
        pred_boxes: np.ndarray,
        letterbox_info: Dict,
    ) -> np.ndarray:
        labels, boxes_xyxy, scores = _decode_raw_outputs(
            pred_logits, pred_boxes, self.target_size, self.num_top_queries
        )
        boxes_orig = _undo_letterbox(boxes_xyxy, letterbox_info)

        keep = scores >= self.conf_threshold
        labels = labels[keep]
        boxes_orig = boxes_orig[keep]
        scores = scores[keep]

        if len(scores) == 0:
            return np.empty((0, 6), dtype=np.float32)

        return np.concatenate(
            [boxes_orig, scores[:, None], labels[:, None].astype(np.float32)],
            axis=1,
        )

    def _postprocess_pytorch(
        self,
        labels: torch.Tensor,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        letterbox_info: Dict,
    ) -> np.ndarray:
        lab = (
            labels[0].detach().cpu().numpy().astype(np.int32)
            if labels.ndim == 2
            else labels.detach().cpu().numpy().astype(np.int32)
        )
        box = (
            boxes[0].detach().cpu().numpy().astype(np.float32)
            if boxes.ndim == 3
            else boxes.detach().cpu().numpy().astype(np.float32)
        )
        sco = (
            scores[0].detach().cpu().numpy().astype(np.float32)
            if scores.ndim == 2
            else scores.detach().cpu().numpy().astype(np.float32)
        )

        box = _undo_letterbox(box, letterbox_info)

        keep = sco >= self.conf_threshold
        lab = lab[keep]
        box = box[keep]
        sco = sco[keep]

        if len(sco) == 0:
            return np.empty((0, 6), dtype=np.float32)

        return np.concatenate(
            [box, sco[:, None], lab[:, None].astype(np.float32)],
            axis=1,
        )

    # ------------------------------------------------------------------
    # Public API  (BaseDetector contract)
    # ------------------------------------------------------------------

    def detect(self, img_bgr: np.ndarray) -> np.ndarray:
        tensor, lb_info = self._preprocess(img_bgr)

        if self.backend == "onnx":
            return self._detect_onnx(tensor, lb_info)
        elif self.backend == "trt":
            return self._detect_trt(tensor, lb_info)
        elif self.backend == "pytorch":
            return self._detect_pytorch(tensor, lb_info, img_bgr.shape[:2])
        else:
            raise RuntimeError(f"Unknown backend: {self.backend}")

    # ------------------------------------------------------------------
    # Backend-specific inference
    # ------------------------------------------------------------------

    def _detect_onnx(self, tensor: np.ndarray, lb_info: Dict) -> np.ndarray:
        if self.fp16:
            tensor = tensor.astype(np.float16)

        outputs = self._ort_session.run(None, {self._onnx_input_name: tensor})
        pred_logits = outputs[0].astype(np.float32)
        pred_boxes = outputs[1].astype(np.float32)

        return self._postprocess(pred_logits, pred_boxes, lb_info)

    def _detect_trt(self, tensor: np.ndarray, lb_info: Dict) -> np.ndarray:
        ctx = self._trt_ctx
        cuda = ctx["cuda"]
        stream = ctx["stream"]

        if self.fp16:
            tensor = tensor.astype(np.float16)

        input_name = next(iter(ctx["host_inputs"]))
        np.copyto(ctx["host_inputs"][input_name], tensor.ravel())
        cuda.memcpy_htod_async(
            ctx["dev_inputs"][input_name],
            ctx["host_inputs"][input_name],
            stream,
        )

        ctx["context"].execute_async_v3(stream_handle=stream.handle)

        pred_logits: Optional[np.ndarray] = None
        pred_boxes: Optional[np.ndarray] = None
        for name, host_buf in ctx["host_outputs"].items():
            cuda.memcpy_dtoh_async(host_buf, ctx["dev_outputs"][name], stream)
        stream.synchronize()

        for name, host_buf in ctx["host_outputs"].items():
            name_lower = name.lower()
            if "logit" in name_lower:
                pred_logits = host_buf.reshape(1, -1, self.num_classes).astype(np.float32)
            elif "box" in name_lower:
                pred_boxes = host_buf.reshape(1, -1, 4).astype(np.float32)

        if pred_logits is None or pred_boxes is None:
            raise RuntimeError(
                "Could not identify pred_logits / pred_boxes from TRT output names: "
                f"{list(ctx['host_outputs'].keys())}"
            )

        return self._postprocess(pred_logits, pred_boxes, lb_info)

    @torch.no_grad()
    def _detect_pytorch(
        self,
        tensor: np.ndarray,
        lb_info: Dict,
        orig_hw: Tuple[int, int],
    ) -> np.ndarray:
        device = torch.device(self.device_str)
        x = torch.from_numpy(tensor).to(device)
        if self.fp16:
            x = x.half()

        eval_h, eval_w = self.target_size
        orig_target_sizes = torch.tensor(
            [[eval_w, eval_h]], device=device, dtype=x.dtype
        )

        labels, boxes, scores = self._model(x, orig_target_sizes)

        return self._postprocess_pytorch(labels, boxes, scores, lb_info)
