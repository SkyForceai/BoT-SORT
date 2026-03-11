"""
D-FINE + BoT-SORT tracking entry point.

Follows the same structure as tools/track_yolov7.py.  Reuses YOLOv7's
LoadImages/LoadStreams data loaders for frame iteration and mc_bot_sort's
BoTSORT tracker.  The only new component is DFINEDetector from dfine/.

Usage (single image folder or video):

    python tools/track_dfine.py /path/to/img_folder \\
        --dfine-backend onnx \\
        --dfine-model /path/to/model.onnx \\
        --num-classes 5 \\
        --conf-thres 0.3 \\
        --device 0


Usage (single video):
    python tools/track_dfine.py "/home/omer.taub/Desktop/original.MPG" \\
        --dfine-backend pytorch \\
        --dfine-model "/home/omer.taub/Desktop/ObjectDetectionResults/TESTfp16_full_5/best_stg2.pth" \\
        --dfine-config "/home/omer.taub/Desktop/ObjectDetectionExperiments/TCB/models/DFINE/configs/dfine_hgnetv2_l_custom.yml" \\
        --tcb-path "/home/omer.taub/Desktop/ObjectDetectionExperiments/TCB" \\
        --num-classes 5 \\
        --target-size 720 1280 \\
        --device 0 \\
        --conf-thres 0.3 \\
        --cmc-method sparseOptFlow \\
        --view-img \\
        --save-frames \\
        --project runs/track_dfine \\
        --name my_video

MOT benchmark evaluation:

    python tools/track_dfine.py /data/MOT17 \\
        --benchmark MOT17 --eval val \\
        --dfine-backend pytorch \\
        --dfine-model /path/to/best_stg1.pth \\
        --dfine-config /path/to/dfine_hgnetv2_l_custom.yml \\
        --tcb-path /path/to/TCB \\
        --num-classes 1 \\
        --target-size 720 1280
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

sys.path.append('.')

from dfine.utils.datasets import LoadImages, LoadStreams
from dfine.utils.general import check_imshow, increment_path, set_logging
from dfine.utils.plots import plot_one_box
from dfine.utils.torch_utils import select_device, time_synchronized

from tracker.mc_bot_sort import BoTSORT
from tracker.tracking_utils.timer import Timer

# D-FINE wrapper (this package)
from dfine import DFINEDetector
from dfine.utils.csv_export import CSVExporter
from dfine.utils.run_info import save_run_info

# ---------------------------------------------------------------------------
# Globals (mirrors track_yolov7.py pattern)
# ---------------------------------------------------------------------------
trackerTimer = Timer()
timer        = Timer()


def detect() -> None:
    """Run detection + tracking for the sequence specified in `opt`."""
    source   = opt.source
    save_img = opt.save_frames and not source.endswith('.txt')
    webcam   = (
        source.isnumeric()
        or source.endswith('.txt')
        or source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    )

    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    save_dir.mkdir(parents=True, exist_ok=True)

    save_run_info(str(save_dir), opt)
    csv_exporter = CSVExporter(str(save_dir))

    set_logging()
    device = select_device(opt.device)

    # ------------------------------------------------------------------ #
    # Build D-FINE detector                                                #
    # ------------------------------------------------------------------ #
    target_size = tuple(opt.target_size)   # (H, W)
    detector = DFINEDetector(
        backend         = opt.dfine_backend,
        model_path      = opt.dfine_model,
        config_path     = opt.dfine_config,
        target_size     = target_size,
        num_classes     = opt.num_classes,
        conf_threshold  = opt.test_conf,
        num_top_queries = 300,
        device          = str(device) if str(device) != 'cpu' else 'cpu',
        fp16            = opt.fp16,
        use_ema         = not opt.no_ema,
        tcb_path        = opt.tcb_path,
    )

    # ------------------------------------------------------------------ #
    # Data loader                                                          #
    # ------------------------------------------------------------------ #
    # Use a fixed stride of 32 (standard for most detectors) to satisfy
    # LoadImages size checks; actual resizing is handled by DFINEDetector.
    stride = 32
    imgsz  = max(target_size)  # used only to satisfy LoadImages API

    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    if opt.ablation:
        dataset.files = dataset.files[len(dataset.files) // 2 + 1:]
        dataset.nf    = len(dataset.files)

    # Class names for visualisation (generic fallback)
    names  = [f'cls{i}' for i in range(opt.num_classes)]
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(100)]

    # ------------------------------------------------------------------ #
    # Tracker                                                              #
    # ------------------------------------------------------------------ #
    tracker = BoTSORT(opt, frame_rate=30.0)

    # ------------------------------------------------------------------ #
    # Inference loop                                                        #
    # ------------------------------------------------------------------ #
    t0      = time.time()
    results = []
    fn      = 0

    for path, img, im0s, vid_cap in dataset:
        fn += 1
        timer.tic()

        # `img` is the pre-scaled image tensor from LoadImages (used only
        # to check webcam batch size); actual inference uses `im0s` (BGR).
        if webcam:
            # im0s is a list of frames (one per stream)
            im0_list  = im0s
            path_list = path
        else:
            im0_list  = [im0s]
            path_list = [path]

        for im0, p in zip(im0_list, path_list):
            # ---- Detection ----
            t1 = time_synchronized()
            detections = detector.detect(im0)   # (N, 6): x1 y1 x2 y2 score cls
            t2 = time_synchronized()
            csv_exporter.add_detections(fn, detections)

            # ---- Tracking ----
            trackerTimer.tic()
            online_targets = tracker.update(detections, im0)
            trackerTimer.toc()
            csv_exporter.add_tracks(fn, online_targets)

            # ---- Collect outputs ----
            for t in online_targets:
                tlwh     = t.tlwh
                tlbr     = t.tlbr
                tid      = t.track_id
                tcls     = t.cls
                vertical = tlwh[2] / tlwh[3] > opt.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                    results.append(
                        f"{fn},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},"
                        f"{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                    )

                    if save_img or opt.view_img:
                        label = (
                            f'{tid}, {int(tcls)}'
                            if opt.hide_labels_name
                            else f'{tid}, {names[int(tcls)] if int(tcls) < len(names) else int(tcls)}'
                        )
                        plot_one_box(
                            tlbr, im0,
                            label=label,
                            color=colors[int(tid) % len(colors)],
                            line_thickness=2,
                        )

            p      = Path(p)
            save_path = str(save_dir / p.name)

            if opt.view_img:
                cv2.imshow('BoT-SORT D-FINE', im0)
                cv2.waitKey(1)

            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()
                        if vid_cap:
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w   = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h   = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h)
                        )
                    vid_writer.write(im0)

        timer.toc()

    csv_exporter.save()

    res_file = opt.project + '/' + opt.name + '.txt'
    with open(res_file, 'w') as f:
        f.writelines(results)
    print(f'save results to {res_file}')
    print(f'Done. ({time.time() - t0:.3f}s)')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='D-FINE + BoT-SORT multi-object tracking'
    )

    # ---- Input / output ----
    parser.add_argument('source', type=str, default='inference/images',
                        help='Source: file/folder/video/stream (0 for webcam)')
    parser.add_argument('--project', default='runs/track',
                        help='Save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='Save results to project/name')
    parser.add_argument('--save-frames', dest='save_frames',
                        default=False, action='store_true',
                        help='Save annotated frames/video')
    parser.add_argument('--view-img', action='store_true',
                        help='Display results live')
    parser.add_argument('--hide-labels-name', default=False, action='store_true',
                        help='Hide class name labels (show track id + class id only)')

    # ---- Benchmark ----
    parser.add_argument('--mot-benchmark', dest='mot_benchmark', default=False,
                        action='store_true',
                        help='Run in MOT benchmark mode (iterate sequences). '
                             'Without this flag, source is used directly.')
    parser.add_argument('--benchmark', dest='benchmark', type=str, default='MOT17',
                        help='Benchmark to evaluate: MOT17 | MOT20')
    parser.add_argument('--eval', dest='split_to_eval', type=str, default='test',
                        help='Split to evaluate: train | val | test')
    parser.add_argument('--ablation', dest='ablation', default=False,
                        action='store_true', help='Ablation mode')
    parser.add_argument('--default-parameters', dest='default_parameters',
                        default=False, action='store_true',
                        help='Use default tracking parameters from the paper')

    # ---- D-FINE specific ----
    parser.add_argument('--dfine-backend', dest='dfine_backend', type=str,
                        default='pytorch', choices=['onnx', 'trt', 'pytorch'],
                        help='D-FINE inference backend')
    parser.add_argument('--dfine-model', dest='dfine_model', type=str, default=None,
                        help='Path to .onnx / .engine / .pth model file')
    parser.add_argument('--dfine-config', dest='dfine_config', type=str, default=None,
                        help='Path to D-FINE YAML config (pytorch backend only)')
    parser.add_argument('--tcb-path', dest='tcb_path', type=str, default=None,
                        help='Absolute path to the TCB repo root (pytorch backend only)')
    parser.add_argument('--target-size', dest='target_size', nargs=2, type=int,
                        default=[720, 1280], metavar=('H', 'W'),
                        help='Letterbox target size H W (default: 720 1280)')
    parser.add_argument('--num-classes', dest='num_classes', type=int, default=5,
                        help='Number of detection classes')
    parser.add_argument('--no-ema', dest='no_ema', default=False, action='store_true',
                        help='Use model weights instead of EMA (pytorch backend)')

    # ---- Detector thresholds ----
    parser.add_argument('--conf-thres', type=float, default=0.01,
                        help='Detection confidence threshold (fed to DFINEDetector)')
    parser.add_argument('--device', default='',
                        help='CUDA device index or "cpu"')
    parser.add_argument('--fp16', default=False, action='store_true',
                        help='FP16 inference')

    # ---- Tracking parameters ----
    parser.add_argument('--track_high_thresh', type=float, default=0.5,
                        help='High-confidence tracking threshold')
    parser.add_argument('--track_low_thresh', default=0.1, type=float,
                        help='Low-confidence tracking threshold')
    parser.add_argument('--new_track_thresh', default=0.6, type=float,
                        help='New track initialisation threshold')
    parser.add_argument('--track_buffer', type=int, default=120,
                        help='Frames to keep a lost track alive')
    parser.add_argument('--match_thresh', type=float, default=0.7,
                        help='IoU matching threshold')
    parser.add_argument('--aspect_ratio_thresh', type=float, default=80,
                        help='Filter boxes with aspect ratio above this value')
    parser.add_argument('--min_box_area', type=float, default=1,
                        help='Filter boxes smaller than this area')
    parser.add_argument('--fuse-score', dest='mot20', default=False,
                        action='store_true',
                        help='Fuse score and IoU for association (MOT20 mode)')

    # ---- CMC ----
    parser.add_argument('--cmc-method', default='none', type=str,
                        help='CMC method: file | sparseOptFlow | orb | ecc | none')

    # ---- ReID ----
    parser.add_argument('--with-reid', dest='with_reid', default=False,
                        action='store_true', help='Enable ReID module')
    parser.add_argument('--fast-reid-config', dest='fast_reid_config',
                        default=r'fast_reid/configs/MOT17/sbs_S50.yml',
                        type=str, help='FastReID config file path')
    parser.add_argument('--fast-reid-weights', dest='fast_reid_weights',
                        default=r'pretrained/mot17_sbs_S50.pth',
                        type=str, help='FastReID weights file path')
    parser.add_argument('--proximity_thresh', type=float, default=0.5,
                        help='Reject ReID matches below this overlap threshold')
    parser.add_argument('--appearance_thresh', type=float, default=0.25,
                        help='Reject ReID matches below this appearance similarity')

    opt = parser.parse_args()
    opt.jde      = False
    opt.exist_ok = True

    # `test_conf` is the pre-filter passed to DFINEDetector; keep it slightly
    # below track_low_thresh so borderline detections are not lost before the
    # tracker's own threshold applies.
    opt.test_conf = max(0.001, opt.track_low_thresh - 0.01)

    print(opt)

    data_path = opt.source
    device    = opt.device

    # ------------------------------------------------------------------ #
    # Decide mode: direct source (video / folder) vs MOT benchmark        #
    # ------------------------------------------------------------------ #
    # If --mot-benchmark is passed, iterate over MOT sequences.
    # Otherwise treat `source` as a direct video file or image folder.

    if not opt.mot_benchmark:
        # ---- Direct mode: single video or image folder ----
        opt.fps        = 30
        opt.batch_size = 1
        opt.trt        = False
        if not hasattr(opt, 'ablation') or opt.ablation is None:
            opt.ablation = False
        Path(opt.project).mkdir(parents=True, exist_ok=True)

        mainTimer = Timer()
        mainTimer.tic()

        with torch.no_grad():
            detect()

        mainTimer.toc()
        print("TOTAL TIME END-to-END (with loading networks and images):",
              mainTimer.total_time)
        if timer.average_time > 0:
            print("TOTAL TIME (Detector + Tracker):",
                  timer.total_time, "FPS:", 1.0 / timer.average_time)
        if trackerTimer.average_time > 0:
            print("TOTAL TIME (Tracker only):",
                  trackerTimer.total_time, "FPS:", 1.0 / trackerTimer.average_time)

    else:
        # ---- MOT benchmark mode ----
        if opt.benchmark == 'MOT20':
            train_seqs = [1, 2, 3, 5]
            test_seqs  = [4, 6, 7, 8]
            seqs_ext   = ['']
            MOT        = 20
        elif opt.benchmark == 'MOT17':
            train_seqs = [2, 4, 5, 9, 10, 11, 13]
            test_seqs  = [1, 3, 6, 7, 8, 12, 14]
            seqs_ext   = ['FRCNN', 'DPM', 'SDP']
            MOT        = 17
        else:
            raise ValueError(f'Unsupported benchmark: {opt.benchmark}')

        ablation = False
        if opt.split_to_eval == 'train':
            seqs = train_seqs
        elif opt.split_to_eval == 'val':
            seqs     = train_seqs
            ablation = True
        elif opt.split_to_eval == 'test':
            seqs = test_seqs
        else:
            raise ValueError(f'Unsupported split: {opt.split_to_eval}')

        mainTimer = Timer()
        mainTimer.tic()

        for ext in seqs_ext:
            for i in seqs:
                seq = f"MOT{MOT}-{'0' if i < 10 else ''}{i}"
                if ext:
                    seq += f'-{ext}'

                opt.name      = seq
                opt.ablation  = ablation
                opt.mot20     = MOT == 20
                opt.fps       = 30
                opt.device    = device
                opt.batch_size = 1
                opt.trt       = False

                split      = 'train' if i in train_seqs else 'test'
                opt.source = f"{data_path}/{split}/{seq}/img1"

                if opt.default_parameters:
                    opt.track_high_thresh = 0.3
                    opt.track_low_thresh  = 0.1
                    opt.track_buffer      = 30
                    opt.new_track_thresh  = opt.track_high_thresh + 0.1

                opt.test_conf = max(0.001, opt.track_low_thresh - 0.01)

                with torch.no_grad():
                    detect()

        mainTimer.toc()
        print("TOTAL TIME END-to-END (with loading networks and images):",
              mainTimer.total_time)
        print("TOTAL TIME (Detector + Tracker):",
              timer.total_time, "FPS:", 1.0 / timer.average_time)
        print("TOTAL TIME (Tracker only):",
              trackerTimer.total_time, "FPS:", 1.0 / trackerTimer.average_time)
