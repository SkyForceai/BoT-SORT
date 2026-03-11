"""
DFINEDetector: wrapper around the D-FINE object detection model for BoT-SORT.

Supports three inference backends:
  - "onnx"    : ONNX Runtime (no TCB dependency)
  - "trt"     : TensorRT engine (no TCB dependency)
  - "pytorch" : PyTorch checkpoint via TCB YAMLConfig + DeployWrapper

All backends expose a common `detect(img_bgr)` interface that returns an
(N, 6) numpy array with columns [x1, y1, x2, y2, score, class_id] in
original-image pixel coordinates, ready to be fed directly to
mc_bot_sort.BoTSORT.update().

Preprocessing and postprocessing are self-contained (no TCB imports needed
for ONNX/TRT backends). The PyTorch backend lazily imports TCB modules from
`tcb_path` at construction time so the import side-effects are isolated.
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ImageNet normalization constants (must match TCB training pipeline)
# ---------------------------------------------------------------------------
_IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_IMG_SCALE = 255.0


# ---------------------------------------------------------------------------
# Standalone preprocessing (no TCB dependency)
# ---------------------------------------------------------------------------

def _letterbox(
    img_rgb: np.ndarray,
    target_size: Tuple[int, int],
    fill: int = 0,
) -> Tuple[np.ndarray, Dict]:
    """
    Aspect-ratio-preserving resize + bottom-right pad.

    Args:
        img_rgb: HWC uint8 RGB image.
        target_size: (H, W) target canvas.
        fill: Padding pixel value (0-255).

    Returns:
        (letterboxed_rgb, info) where info has keys:
            orig_size   : (H, W) original
            resized_size: (H, W) after scale, before pad
            padding     : [pad_left, pad_top, pad_right, pad_bottom]
            scale       : float uniform scale factor
    """
    target_h, target_w = target_size
    orig_h, orig_w = img_rgb.shape[:2]

    scale = min(target_h / orig_h, target_w / orig_w)
    resized_h = max(1, int(math.floor(orig_h * scale)))
    resized_w = max(1, int(math.floor(orig_w * scale)))

    # Use cv2 for speed; import locally to avoid hard dependency at module level
    import cv2
    resized = cv2.resize(img_rgb, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((target_h, target_w, 3), fill, dtype=np.uint8)
    canvas[:resized_h, :resized_w] = resized

    pad_left, pad_top = 0, 0
    pad_right  = target_w - resized_w
    pad_bottom = target_h - resized_h

    info = {
        "orig_size":    (orig_h, orig_w),
        "resized_size": (resized_h, resized_w),
        "padding":      [pad_left, pad_top, pad_right, pad_bottom],
        "scale":        scale,
    }
    return canvas, info


def _normalize_to_nchw(img_rgb: np.ndarray) -> np.ndarray:
    """
    Scale by 1/255, apply ImageNet mean/std, return NCHW float32 array.

    Args:
        img_rgb: (H, W, 3) uint8 RGB image.

    Returns:
        (1, 3, H, W) float32 numpy array.
    """
    img = img_rgb.astype(np.float32) / _IMG_SCALE
    img = (img - _IMG_MEAN) / _IMG_STD          # HWC
    img = img.transpose(2, 0, 1)[np.newaxis]    # 1CHW
    return np.ascontiguousarray(img, dtype=np.float32)


# ---------------------------------------------------------------------------
# Standalone postprocessing (no TCB dependency)
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))


def _decode_raw_outputs(
    pred_logits: np.ndarray,
    pred_boxes:  np.ndarray,
    target_size: Tuple[int, int],
    num_top:     int = 300,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert raw D-FINE model outputs to scored xyxy detections.

    Args:
        pred_logits: (num_queries, num_classes)  or  (1, num_queries, num_classes)
        pred_boxes:  (num_queries, 4)            or  (1, num_queries, 4)
                     boxes in normalized cxcywh [0,1]
        target_size: (H, W) of the letterboxed canvas (for pixel conversion)
        num_top:     maximum number of detections to return

    Returns:
        labels:    (N,) int32
        boxes_xyxy:(N, 4) float32  xyxy in letterbox pixel coords
        scores:    (N,) float32
    """
    # Strip batch dim if present
    if pred_logits.ndim == 3:
        pred_logits = pred_logits[0]
    if pred_boxes.ndim == 3:
        pred_boxes = pred_boxes[0]

    scores_all = _sigmoid(pred_logits)           # (Q, C)
    flat_scores = scores_all.flatten()

    num_top_actual = min(num_top, len(flat_scores))
    top_idx = np.argpartition(flat_scores, -num_top_actual)[-num_top_actual:]
    top_idx = top_idx[np.argsort(flat_scores[top_idx])[::-1]]

    n_classes     = scores_all.shape[1]
    query_indices = top_idx // n_classes
    class_indices = top_idx  % n_classes

    scores = flat_scores[top_idx].astype(np.float32)
    labels = class_indices.astype(np.int32)

    boxes_cxcywh = pred_boxes[query_indices]     # (N, 4)
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


def _undo_letterbox(
    boxes:          np.ndarray,
    letterbox_info: Dict,
) -> np.ndarray:
    """
    Map xyxy boxes from letterbox-canvas space to original-image space.

    Args:
        boxes:          (N, 4) float32 xyxy in letterbox pixel coords.
        letterbox_info: dict returned by _letterbox().

    Returns:
        (N, 4) float32 xyxy in original-image pixel coords (clipped).
    """
    if boxes.size == 0:
        return boxes

    boxes = boxes.copy()
    pad_left, pad_top = letterbox_info["padding"][0], letterbox_info["padding"][1]
    scale  = letterbox_info["scale"]
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

class DFINEDetector:
    """
    Thin wrapper around D-FINE detection for BoT-SORT tracking.

    Usage::

        detector = DFINEDetector(
            backend="onnx",
            model_path="/path/to/model.onnx",
            target_size=(720, 1280),
            conf_threshold=0.3,
        )

        # img_bgr: HxWx3 uint8 numpy array (OpenCV frame)
        detections = detector.detect(img_bgr)
        # detections: (N, 6) [x1, y1, x2, y2, score, class_id]

        online_targets = tracker.update(detections, img_bgr)
    """

    BACKENDS: List[str] = ["onnx", "trt", "pytorch"]

    def __init__(
        self,
        backend:         str              = "onnx",
        model_path:      Optional[str]    = None,
        config_path:     Optional[str]    = None,
        target_size:     Tuple[int, int]  = (720, 1280),
        num_classes:     int              = 5,
        conf_threshold:  float            = 0.3,
        num_top_queries: int              = 300,
        device:          str              = "cuda:0",
        fp16:            bool             = False,
        use_ema:         bool             = True,
        tcb_path:        Optional[str]    = None,
    ) -> None:
        """
        Args:
            backend:         One of "onnx", "trt", "pytorch".
            model_path:      Path to .onnx / .engine / .pth file.
            config_path:     YAML config path (required for pytorch backend).
            target_size:     (H, W) letterbox canvas size.
            num_classes:     Number of detection classes.
            conf_threshold:  Minimum confidence to keep a detection.
            num_top_queries: Top-k detections from D-FINE head.
            device:          Inference device string, e.g. "cuda:0" or "cpu".
            fp16:            Run in FP16 precision (ONNX/TRT/PyTorch).
            use_ema:         Prefer EMA weights when loading a .pth checkpoint.
            tcb_path:        Absolute path to the TCB repo root (pytorch only).
        """
        if backend not in self.BACKENDS:
            raise ValueError(f"backend must be one of {self.BACKENDS}, got '{backend}'")

        self.backend         = backend
        self.model_path      = str(model_path) if model_path else None
        self.config_path     = str(config_path) if config_path else None
        self.target_size     = target_size          # (H, W)
        self.num_classes     = num_classes
        self.conf_threshold  = conf_threshold
        self.num_top_queries = num_top_queries
        self.device_str      = device
        self.fp16            = fp16
        self.use_ema         = use_ema
        self.tcb_path        = str(tcb_path) if tcb_path else None

        self._model      = None   # backend-specific handle
        self._ort_session = None  # ONNX Runtime session
        self._trt_ctx    = None   # TRT execution context + buffers

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
        """Load TensorRT engine and create execution context + pinned I/O buffers."""
        try:
            import tensorrt as trt
            import pycuda.autoinit  # noqa: F401  (side-effect: init CUDA context)
            import pycuda.driver as cuda
        except ImportError as e:
            raise ImportError(
                "tensorrt and pycuda are required for the TRT backend. "
                "Install pycuda with: pip install pycuda"
            ) from e

        if not self.model_path:
            raise ValueError("model_path must be provided for the TRT backend.")

        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime    = trt.Runtime(trt_logger)

        with open(self.model_path, "rb") as f:
            engine_data = f.read()

        engine  = runtime.deserialize_cuda_engine(engine_data)
        context = engine.create_execution_context()

        # Allocate host (pinned) + device buffers for every I/O tensor.
        # TRT 10+ uses set_tensor_address / execute_async_v3 (the old
        # execute_async_v2 with a flat bindings list was removed).
        host_inputs:  Dict[str, np.ndarray] = {}
        host_outputs: Dict[str, np.ndarray] = {}
        dev_inputs:   Dict[str, object]     = {}
        dev_outputs:  Dict[str, object]     = {}
        tensor_shapes: Dict[str, tuple]     = {}

        for i in range(engine.num_io_tensors):
            name  = engine.get_tensor_name(i)
            shape = tuple(engine.get_tensor_shape(name))
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            size  = int(np.prod(shape)) * np.dtype(dtype).itemsize

            host_buf = cuda.pagelocked_empty(int(np.prod(shape)), dtype)
            dev_buf  = cuda.mem_alloc(size)
            tensor_shapes[name] = shape

            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                host_inputs[name]  = host_buf
                dev_inputs[name]   = dev_buf
            else:
                host_outputs[name] = host_buf
                dev_outputs[name]  = dev_buf

            # Register device pointer with the execution context (v3 API)
            context.set_tensor_address(name, int(dev_buf))

        self._trt_ctx = {
            "engine":        engine,
            "context":       context,
            "stream":        cuda.Stream(),
            "host_inputs":   host_inputs,
            "host_outputs":  host_outputs,
            "dev_inputs":    dev_inputs,
            "dev_outputs":   dev_outputs,
            "tensor_shapes": tensor_shapes,
            "cuda":          cuda,
        }
        logger.info("D-FINE TRT engine loaded from %s", self.model_path)

    def _load_pytorch(self) -> None:
        """
        Load D-FINE via TCB's YAMLConfig + DeployWrapper.

        Requires tcb_path pointing at the TCB repo root, and config_path
        pointing at a D-FINE YAML config.
        """
        if not self.tcb_path:
            raise ValueError("tcb_path must be provided for the pytorch backend.")
        if not self.config_path:
            raise ValueError("config_path must be provided for the pytorch backend.")

        tcb = Path(self.tcb_path)
        if not tcb.exists():
            raise FileNotFoundError(f"TCB path not found: {tcb}")

        import importlib
        import os
        import torch.nn as nn

        tcb_str = str(tcb)

        # ---- Isolate TCB imports from the rest of sys.path ----
        # Other packages on sys.path (e.g. yolov7/) have their own
        # top-level `models` and `data` packages that collide with TCB's.
        # TCB also uses implicit namespace packages (models/ and
        # models/DFINE/ lack __init__.py), which breaks when multiple
        # paths expose the same namespace.  We work around this by:
        #   1. Evicting conflicting modules from sys.modules
        #   2. Temporarily making TCB the only path entry (+ stdlib)
        #   3. Ensuring missing __init__.py stubs exist
        #   4. Doing all TCB imports
        #   5. Restoring everything afterwards
        saved_cwd  = os.getcwd()
        saved_path = sys.path.copy()

        # Stash any already-imported modules whose names collide with TCB's
        _tcb_top_level = ("models", "data", "core", "training")
        stashed_modules: Dict[str, object] = {}
        for prefix in _tcb_top_level:
            for key in list(sys.modules):
                if key == prefix or key.startswith(prefix + "."):
                    stashed_modules[key] = sys.modules.pop(key)

        # Ensure TCB has proper regular-package __init__.py files so the
        # import system doesn't fall back to namespace-package resolution.
        _init_stubs_created: list = []
        for sub in ("models", "models/DFINE"):
            init_file = tcb / sub / "__init__.py"
            if not init_file.exists():
                init_file.write_text("")
                _init_stubs_created.append(init_file)

        try:
            # Temporarily trim sys.path to TCB + stdlib only, so that
            # yolov7/models/ can never shadow TCB/models/.
            stdlib_paths = [p for p in saved_path
                           if "site-packages" in p
                           or "lib/python" in p
                           or p == ""]
            sys.path = [tcb_str] + stdlib_paths

            os.chdir(tcb_str)
            importlib.invalidate_caches()

            # Side-effect imports that populate TCB's workspace registry
            for mod in ("data", "training.optim"):
                try:
                    importlib.import_module(mod)
                except Exception:
                    pass

            # Register the D-FINE model modules
            importlib.import_module("models.DFINE.nn")

            from core import YAMLConfig  # type: ignore[import]
        finally:
            os.chdir(saved_cwd)
            sys.path = saved_path

        # Build config; disable backbone pretrained download
        update_dict = {"num_classes": self.num_classes}
        cfg = YAMLConfig(self.config_path, **update_dict)
        for name in ("HGNetv2", "DFINEHGNetv2", "DomeHGNetv2"):
            if name in cfg.yaml_cfg and isinstance(cfg.yaml_cfg[name], dict):
                cfg.yaml_cfg[name]["pretrained"] = False

        # Load checkpoint weights
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
                self_.model     = model_.deploy() if hasattr(model_, "deploy") else model_.eval()
                self_.postproc  = (
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
                        return (res["labels"].unsqueeze(0),
                                res["boxes"].unsqueeze(0),
                                res["scores"].unsqueeze(0))
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

        # Strip "module." prefix from DataParallel/DDP checkpoints
        return {
            (k[7:] if k.startswith("module.") else k): v
            for k, v in state.items()
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preprocess(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Convert a BGR frame to a normalised NCHW float32 array suitable for
        D-FINE inference.

        Args:
            img_bgr: (H, W, 3) uint8 BGR image (standard OpenCV format).

        Returns:
            tensor:         (1, 3, H_lb, W_lb) float32 numpy array.
            letterbox_info: dict with keys orig_size, resized_size, padding, scale.
        """
        import cv2
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        letterboxed, lb_info = _letterbox(img_rgb, self.target_size)
        tensor = _normalize_to_nchw(letterboxed)
        return tensor, lb_info

    @torch.no_grad()
    def postprocess(
        self,
        pred_logits:    np.ndarray,
        pred_boxes:     np.ndarray,
        letterbox_info: Dict,
    ) -> np.ndarray:
        """
        Convert raw D-FINE outputs to tracker-ready detections.

        Works with the raw ONNX/TRT outputs (pred_logits, pred_boxes in
        cxcywh normalized space).  The pytorch backend has its own path
        that calls this only for the letterbox-undo step.

        Args:
            pred_logits:    (1, Q, C) or (Q, C) float32 – raw class logits.
            pred_boxes:     (1, Q, 4) or (Q, 4) float32 – boxes in cxcywh [0,1].
            letterbox_info: dict from preprocess().

        Returns:
            (N, 6) float32 array: [x1, y1, x2, y2, score, class_id]
            Rows are in descending score order; only rows with
            score >= conf_threshold are included.
        """
        labels, boxes_xyxy, scores = _decode_raw_outputs(
            pred_logits, pred_boxes, self.target_size, self.num_top_queries
        )
        boxes_orig = _undo_letterbox(boxes_xyxy, letterbox_info)

        keep = scores >= self.conf_threshold
        labels     = labels[keep]
        boxes_orig = boxes_orig[keep]
        scores     = scores[keep]

        if len(scores) == 0:
            return np.empty((0, 6), dtype=np.float32)

        detections = np.concatenate(
            [boxes_orig, scores[:, None], labels[:, None].astype(np.float32)],
            axis=1,
        )
        return detections

    def _postprocess_pytorch(
        self,
        labels:         torch.Tensor,
        boxes:          torch.Tensor,
        scores:         torch.Tensor,
        letterbox_info: Dict,
    ) -> np.ndarray:
        """
        Convert pytorch DeployWrapper outputs to tracker format.

        The pytorch postprocessor already scales boxes to the eval canvas
        (target_size) in xyxy pixel coords.  We only need to undo the
        letterbox transform and apply the confidence filter.

        Args:
            labels:  (1, Q) or (Q,) int tensor
            boxes:   (1, Q, 4) or (Q, 4) xyxy pixel coords on letterbox canvas
            scores:  (1, Q) or (Q,) float tensor
            letterbox_info: dict from preprocess()

        Returns:
            (N, 6) float32 array: [x1, y1, x2, y2, score, class_id]
        """
        # Strip batch dim; .detach() in case grad is still tracked
        lab = labels[0].detach().cpu().numpy().astype(np.int32)   if labels.ndim  == 2 else labels.detach().cpu().numpy().astype(np.int32)
        box = boxes[0].detach().cpu().numpy().astype(np.float32)  if boxes.ndim   == 3 else boxes.detach().cpu().numpy().astype(np.float32)
        sco = scores[0].detach().cpu().numpy().astype(np.float32) if scores.ndim  == 2 else scores.detach().cpu().numpy().astype(np.float32)

        box = _undo_letterbox(box, letterbox_info)

        keep = sco >= self.conf_threshold
        lab  = lab[keep]
        box  = box[keep]
        sco  = sco[keep]

        if len(sco) == 0:
            return np.empty((0, 6), dtype=np.float32)

        detections = np.concatenate(
            [box, sco[:, None], lab[:, None].astype(np.float32)],
            axis=1,
        )
        return detections

    def detect(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Run D-FINE detection on a single BGR frame.

        Args:
            img_bgr: (H, W, 3) uint8 numpy array (OpenCV BGR format).

        Returns:
            (N, 6) float32 numpy array: [x1, y1, x2, y2, score, class_id]
            in original-image pixel coordinates.  Returns an empty (0, 6)
            array when there are no detections above conf_threshold.
        """
        tensor, lb_info = self.preprocess(img_bgr)

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

        outputs = self._ort_session.run(
            None, {self._onnx_input_name: tensor}
        )
        # Outputs: [pred_logits (1,Q,C), pred_boxes (1,Q,4)]
        pred_logits = outputs[0].astype(np.float32)
        pred_boxes  = outputs[1].astype(np.float32)

        return self.postprocess(pred_logits, pred_boxes, lb_info)

    def _detect_trt(self, tensor: np.ndarray, lb_info: Dict) -> np.ndarray:
        ctx    = self._trt_ctx
        cuda   = ctx["cuda"]
        stream = ctx["stream"]

        if self.fp16:
            tensor = tensor.astype(np.float16)

        # Copy input to pinned host buffer and upload to device
        input_name = next(iter(ctx["host_inputs"]))
        np.copyto(ctx["host_inputs"][input_name], tensor.ravel())
        cuda.memcpy_htod_async(
            ctx["dev_inputs"][input_name],
            ctx["host_inputs"][input_name],
            stream,
        )

        # Execute (TRT 10+ v3 API -- addresses already set in _load_trt)
        ctx["context"].execute_async_v3(stream_handle=stream.handle)

        # Download outputs
        pred_logits: Optional[np.ndarray] = None
        pred_boxes:  Optional[np.ndarray] = None
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

        return self.postprocess(pred_logits, pred_boxes, lb_info)

    def _detect_pytorch(
        self,
        tensor:   np.ndarray,
        lb_info:  Dict,
        orig_hw:  Tuple[int, int],
    ) -> np.ndarray:
        device = torch.device(self.device_str)
        x = torch.from_numpy(tensor).to(device)
        if self.fp16:
            x = x.half()

        # Pass eval-canvas size so the postprocessor scales boxes to canvas coords
        eval_h, eval_w = self.target_size
        orig_target_sizes = torch.tensor([[eval_w, eval_h]], device=device, dtype=x.dtype)

        labels, boxes, scores = self._model(x, orig_target_sizes)

        return self._postprocess_pytorch(labels, boxes, scores, lb_info)
