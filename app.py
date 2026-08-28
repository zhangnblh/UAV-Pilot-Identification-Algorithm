from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
from mmpose.apis import MMPoseInferencer

from pose_rules_2 import (
    ByteTrackLite,
    DetectionQualityConfig,
    LEFT_EAR,
    LEFT_ELBOW,
    LEFT_EYE,
    LEFT_HIP,
    LEFT_SHOULDER,
    LEFT_WRIST,
    NECK,
    NOSE,
    PilotStateMachine,
    RIGHT_EAR,
    RIGHT_ELBOW,
    RIGHT_EYE,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    RuleConfig,
    analyze_pose,
    evaluate_detection_quality,
)


SKELETON = (
    (LEFT_EAR, LEFT_EYE),
    (LEFT_EYE, NOSE),
    (NOSE, RIGHT_EYE),
    (RIGHT_EYE, RIGHT_EAR),
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW),
    (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW),
    (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, NECK),
    (RIGHT_SHOULDER, NECK),
    (LEFT_SHOULDER, LEFT_HIP),
    (RIGHT_SHOULDER, RIGHT_HIP),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mark persistent two-hand-operation pilot candidates."
    )
    parser.add_argument(
        "--input",
        default="0",
        help="Camera index, local video path, or RTSP/HTTP stream URL.",
    )
    parser.add_argument("--output", help="Optional H.264 annotated MP4 output path.")
    parser.add_argument(
        "--rtsp-output",
        help="Optional RTSP publish URL, for example rtsp://127.0.0.1:8554/pilot.",
    )
    parser.add_argument(
        "--ffmpeg",
        default="ffmpeg",
        help="FFmpeg executable used for H.264 file/RTSP output.",
    )
    parser.add_argument("--encoder-crf", type=int, default=20)
    parser.add_argument(
        "--encoder-bitrate",
        default="3M",
        help="H.264 maximum bitrate and VBV buffer size, e.g. 3M.",
    )
    parser.add_argument("--encoder-preset", default="veryfast")
    parser.add_argument(
        "--output-fps",
        type=float,
        default=0.0,
        help="0 uses input FPS. Live outputs repeat the complete last annotated frame.",
    )
    latest_group = parser.add_mutually_exclusive_group()
    latest_group.add_argument(
        "--latest-frame",
        dest="latest_frame",
        action="store_true",
        help="Continuously discard stale captured frames (default for live sources).",
    )
    latest_group.add_argument(
        "--no-latest-frame",
        dest="latest_frame",
        action="store_false",
        help="Process every captured frame, including buffered live frames.",
    )
    parser.set_defaults(latest_frame=None)
    parser.add_argument("--jsonl", help="Optional per-frame candidate JSONL output path.")
    parser.add_argument("--pose-model", default="body26", help="MMPose model alias.")
    parser.add_argument(
        "--cache-dir",
        default=str(Path(__file__).resolve().parent / ".cache"),
        help="Writable directory for downloaded model weights.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda:0, etc.",
    )
    parser.add_argument("--no-show", action="store_true", help="Disable preview window.")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means unlimited.")
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help=(
            "Offline files only: run inference on every Nth source frame while "
            "preserving original frame indexes and timestamps."
        ),
    )
    parser.add_argument("--keypoint-thr", type=float, default=0.35)
    parser.add_argument("--holding-thr", type=float, default=0.80)
    parser.add_argument(
        "--head-pitch-thr",
        type=float,
        default=0.42,
        help="Nose displacement relative to the eye line; auxiliary evidence only.",
    )
    parser.add_argument(
        "--head-neck-max",
        type=float,
        default=0.90,
        help="Maximum nose-neck distance / shoulder width for a plausible down pose.",
    )
    parser.add_argument("--history", type=int, default=12)
    parser.add_argument("--min-positive", type=int, default=8)
    parser.add_argument(
        "--candidate-hold-seconds",
        type=float,
        default=5.0,
        help="Keep candidate state this long through head-up or missing pose evidence.",
    )
    parser.add_argument("--track-iou", type=float, default=0.15)
    parser.add_argument(
        "--track-center-distance",
        type=float,
        default=0.80,
        help=(
            "Maximum normalized center distance used for track association. "
            "Smaller values reduce ID switches when people cross."
        ),
    )
    parser.add_argument("--track-high-score", type=float, default=0.50)
    parser.add_argument("--track-low-score", type=float, default=0.10)
    parser.add_argument("--track-max-age", type=int, default=30)
    parser.add_argument("--max-box-area-ratio", type=float, default=0.55)
    parser.add_argument("--min-box-area-ratio", type=float, default=0.0003)
    parser.add_argument("--min-box-height", type=float, default=24.0)
    parser.add_argument("--min-person-aspect", type=float, default=0.12)
    parser.add_argument("--max-person-aspect", type=float, default=1.20)
    parser.add_argument("--min-valid-keypoints", type=int, default=6)
    parser.add_argument(
        "--roi-polygon",
        default="",
        help=(
            "Optional walking-area polygon: 'x1,y1;x2,y2;...'. "
            "Use either pixel coordinates or normalized 0..1 coordinates."
        ),
    )
    parser.add_argument("--draw-keypoints", action="store_true")
    parser.add_argument(
        "--controller-candidate-mode",
        choices=("all", "broad", "strict"),
        default="broad",
        help=(
            "Which tracked people should be offered to the downstream controller "
            "detector. broad keeps weak/observing poses for high recall."
        ),
    )
    return parser.parse_args()


def resolve_source(value: str) -> Any:
    stripped = value.strip()
    if stripped.isdigit() and not Path(stripped).exists():
        return int(stripped)
    return stripped


def is_live_source(source: Any) -> bool:
    if isinstance(source, int):
        return True
    value = str(source).lower()
    return value.startswith(("rtsp://", "rtsps://", "http://", "https://", "udp://"))


def open_capture(source: Any, low_latency: bool) -> cv2.VideoCapture:
    is_rtsp = isinstance(source, str) and source.lower().startswith(("rtsp://", "rtsps://"))
    if low_latency and is_rtsp:
        # OpenCV passes these options to its FFmpeg backend. Some builds ignore
        # CAP_PROP_BUFFERSIZE, so stale-frame dropping is still handled below.
        os.environ.setdefault(
            "OPENCV_FFMPEG_CAPTURE_OPTIONS",
            "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0",
        )
    backend = cv2.CAP_FFMPEG if is_rtsp else cv2.CAP_ANY
    capture = cv2.VideoCapture(source, backend)
    if low_latency:
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


@dataclass(frozen=True)
class CapturedFrame:
    sequence: int
    captured_at: float
    image: np.ndarray


class LatestFrameReader:
    """Continuously decode and expose only the newest frame.

    Local files can be paced at their declared FPS when they are used as a
    real-time RTSP source; network streams are already paced by their sender.
    """

    def __init__(self, capture: cv2.VideoCapture, pace_fps: float = 0.0):
        self.capture = capture
        self.pace_fps = pace_fps
        self._condition = threading.Condition()
        self._latest: Optional[CapturedFrame] = None
        self._ended = False
        self._stop = False
        self._thread = threading.Thread(
            target=self._run, name="latest-frame-reader", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        sequence = 0
        started = time.monotonic()
        while not self._stop:
            ok, frame = self.capture.read()
            if not ok:
                break
            sequence += 1
            if self.pace_fps > 0:
                target_time = started + (sequence - 1) / self.pace_fps
                delay = target_time - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                if self._stop:
                    break
            packet = CapturedFrame(sequence, time.monotonic(), frame)
            with self._condition:
                self._latest = packet
                self._condition.notify_all()
        with self._condition:
            self._ended = True
            self._condition.notify_all()

    def read_after(
        self, sequence: int, timeout: float = 5.0
    ) -> Optional[CapturedFrame]:
        deadline = time.monotonic() + timeout
        with self._condition:
            while (
                not self._ended
                and (self._latest is None or self._latest.sequence <= sequence)
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)
            if self._latest is None or self._latest.sequence <= sequence:
                return None
            return self._latest

    def close(self) -> None:
        self._stop = True
        self.capture.release()
        with self._condition:
            self._condition.notify_all()
        self._thread.join(timeout=2.0)

    @property
    def ended(self) -> bool:
        with self._condition:
            return self._ended


class FFmpegH264Writer:
    """Encode raw BGR frames once, either to MP4 or directly to RTSP."""

    def __init__(
        self,
        target: str,
        fps: float,
        size: Tuple[int, int],
        ffmpeg: str,
        crf: int,
        bitrate: str,
        preset: str,
        rtsp: bool = False,
    ):
        executable = shutil.which(ffmpeg)
        if executable is None:
            raise RuntimeError(
                f"FFmpeg executable not found: {ffmpeg!r}. Install FFmpeg or use --ffmpeg."
            )
        width, height = size
        gop = max(1, int(round(fps)))
        command = [
            executable,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            f"{fps:.6f}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-tune",
            "zerolatency",
            "-crf",
            str(crf),
            "-maxrate",
            bitrate,
            "-bufsize",
            bitrate,
            "-g",
            str(gop),
            "-keyint_min",
            str(gop),
            "-sc_threshold",
            "0",
            "-bf",
            "0",
            "-pix_fmt",
            "yuv420p",
        ]
        if rtsp:
            command += ["-rtsp_transport", "tcp", "-f", "rtsp", target]
        else:
            output_path = Path(target)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            command += ["-movflags", "+faststart", "-y", str(output_path)]
        self.target = target
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
        )

    def write(self, frame: np.ndarray) -> None:
        if self._process.stdin is None or self._process.poll() is not None:
            raise RuntimeError(f"FFmpeg output stopped unexpectedly: {self.target}")
        try:
            self._process.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError as error:
            raise RuntimeError(f"FFmpeg output pipe closed: {self.target}") from error

    def release(self) -> None:
        if self._process.stdin is not None and not self._process.stdin.closed:
            try:
                self._process.stdin.close()
            except BrokenPipeError:
                pass
        try:
            return_code = self._process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self._process.kill()
            return_code = self._process.wait()
        if return_code != 0:
            raise RuntimeError(
                f"FFmpeg exited with code {return_code}: {self.target}"
            )


class RealtimeFramePublisher:
    """Publish at a stable FPS while repeating the whole annotated frame.

    Repeating the complete frame keeps pixels and boxes synchronized. It never
    paints an old detection result over a newer, unrelated camera frame.
    """

    def __init__(self, writer: FFmpegH264Writer, fps: float):
        self.writer = writer
        self.period = 1.0 / fps
        self._condition = threading.Condition()
        self._latest: Optional[np.ndarray] = None
        self._stop = False
        self._error: Optional[BaseException] = None
        self._thread = threading.Thread(
            target=self._run, name="realtime-frame-publisher", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        next_deadline = time.monotonic()
        try:
            while True:
                with self._condition:
                    while self._latest is None and not self._stop:
                        self._condition.wait()
                    if self._stop:
                        break
                    frame = self._latest
                now = time.monotonic()
                if now < next_deadline:
                    time.sleep(next_deadline - now)
                self.writer.write(frame)
                next_deadline += self.period
                if next_deadline < time.monotonic() - self.period:
                    next_deadline = time.monotonic()
        except BaseException as error:
            self._error = error

    def update(self, frame: np.ndarray) -> None:
        if self._error is not None:
            raise RuntimeError("Realtime output failed") from self._error
        with self._condition:
            self._latest = frame
            self._condition.notify_all()

    def release(self) -> None:
        with self._condition:
            self._stop = True
            self._condition.notify_all()
        self._thread.join(timeout=5.0)
        self.writer.release()
        if self._error is not None:
            raise RuntimeError("Realtime output failed") from self._error


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _unwrap_predictions(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    predictions: Any = result.get("predictions", [])
    while (
        isinstance(predictions, list)
        and len(predictions) == 1
        and isinstance(predictions[0], list)
    ):
        predictions = predictions[0]
    if isinstance(predictions, dict):
        predictions = [predictions]
    return [item for item in predictions if isinstance(item, dict)]


def _instance_arrays(
    instance: Dict[str, Any],
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    if "keypoints" not in instance:
        return None
    keypoints = np.asarray(instance["keypoints"], dtype=np.float32)
    while keypoints.ndim > 2 and len(keypoints) == 1:
        keypoints = keypoints[0]
    if keypoints.ndim != 2 or keypoints.shape[1] < 2:
        return None

    raw_scores = instance.get("keypoint_scores")
    if raw_scores is None and keypoints.shape[1] >= 3:
        scores = keypoints[:, 2]
    elif raw_scores is None:
        scores = np.ones(len(keypoints), dtype=np.float32)
    else:
        scores = np.asarray(raw_scores, dtype=np.float32).reshape(-1)

    raw_bbox = instance.get("bbox")
    if raw_bbox is not None:
        bbox = np.asarray(raw_bbox, dtype=np.float32).reshape(-1)[:4]
    else:
        confident = keypoints[scores[: len(keypoints)] >= 0.2, :2]
        if not len(confident):
            return None
        x1, y1 = confident.min(axis=0)
        x2, y2 = confident.max(axis=0)
        pad_x = max(10.0, float(x2 - x1) * 0.25)
        pad_y = max(10.0, float(y2 - y1) * 0.15)
        bbox = np.asarray([x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y])

    raw_detection_score = instance.get(
        "bbox_score",
        instance.get("bbox_scores", instance.get("detection_score")),
    )
    if raw_detection_score is None:
        # Some inferencer versions omit the detector score. Keep an explicit
        # upper-body proxy. Averaging all Body26 points penalizes otherwise valid
        # people whose legs are cropped or occluded and can prevent track birth.
        core_indices = (
            LEFT_SHOULDER,
            RIGHT_SHOULDER,
            LEFT_ELBOW,
            RIGHT_ELBOW,
            LEFT_WRIST,
            RIGHT_WRIST,
        )
        core_scores = [
            float(scores[index])
            for index in core_indices
            if index < len(scores) and np.isfinite(scores[index])
        ]
        finite_scores = scores[np.isfinite(scores)]
        detection_score = float(
            np.mean(core_scores)
            if core_scores
            else np.mean(finite_scores)
            if len(finite_scores)
            else 0.0
        )
    else:
        flattened_score = np.asarray(raw_detection_score, dtype=np.float32).reshape(-1)
        detection_score = float(flattened_score[0]) if len(flattened_score) else 0.0
    return keypoints[:, :2], scores, bbox, detection_score


def run_pose(
    inferencer: MMPoseInferencer, frame: np.ndarray
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    result = next(inferencer(frame, return_vis=False, show=False))
    people = []
    for instance in _unwrap_predictions(result):
        arrays = _instance_arrays(instance)
        if arrays is not None:
            people.append(arrays)
    return people


def clip_bbox_xyxy(
    bbox: np.ndarray, frame_width: int, frame_height: int
) -> np.ndarray:
    """Clip an xyxy person box before tracking, analysis, and JSON export."""
    box = np.asarray(bbox, dtype=np.float32).reshape(-1)[:4].copy()
    box[[0, 2]] = np.clip(box[[0, 2]], 0, frame_width)
    box[[1, 3]] = np.clip(box[[1, 3]], 0, frame_height)
    return box


def controller_roi_candidate(features: Any, state: str, mode: str) -> bool:
    """High-recall gate for the downstream controller detector."""
    if mode == "all":
        # Reaching this function already means the person passed detection
        # quality filtering and received a track ID. Pose geometry may still be
        # invalid because a wrist is occluded; all mode must keep that person.
        return True
    if mode == "strict":
        return bool(features.frame_candidate or state == "candidate")
    return bool(
        state in {"observing", "candidate"}
        or (
            features.valid
            and (
            features.strong_candidate
            or features.weak_candidate
            or features.grip_geometry_valid
            )
        )
    )


def serialize_keypoints(
    keypoints: np.ndarray, scores: np.ndarray
) -> List[List[float]]:
    """Serialize Body26 points as compact [x, y, score] triplets."""
    count = min(len(keypoints), len(scores))
    serialized: List[List[float]] = []
    for index in range(count):
        x = float(keypoints[index, 0])
        y = float(keypoints[index, 1])
        score = float(scores[index])
        if not np.all(np.isfinite([x, y, score])):
            x, y, score = 0.0, 0.0, 0.0
        serialized.append([round(x, 2), round(y, 2), round(score, 4)])
    return serialized


def parse_roi_polygon(
    value: str, frame_width: int, frame_height: int
) -> Optional[np.ndarray]:
    if not value.strip():
        return None
    try:
        points = [
            (float(pair.split(",")[0]), float(pair.split(",")[1]))
            for pair in value.split(";")
            if pair.strip()
        ]
    except (ValueError, IndexError) as error:
        raise ValueError(
            "--roi-polygon must look like 'x1,y1;x2,y2;x3,y3'"
        ) from error
    if len(points) < 3:
        raise ValueError("--roi-polygon needs at least three points")
    polygon = np.asarray(points, dtype=np.float32)
    if np.all((polygon >= 0.0) & (polygon <= 1.0)):
        polygon[:, 0] *= frame_width
        polygon[:, 1] *= frame_height
    return polygon


def draw_pose(frame: np.ndarray, keypoints: np.ndarray, scores: np.ndarray, threshold: float) -> None:
    for start, end in SKELETON:
        if start >= len(scores) or end >= len(scores):
            continue
        if scores[start] < threshold or scores[end] < threshold:
            continue
        a = tuple(np.round(keypoints[start]).astype(int))
        b = tuple(np.round(keypoints[end]).astype(int))
        cv2.line(frame, a, b, (255, 200, 0), 2, cv2.LINE_AA)
    for index in {item for edge in SKELETON for item in edge}:
        if index < len(scores) and scores[index] >= threshold:
            point = tuple(np.round(keypoints[index]).astype(int))
            cv2.circle(frame, point, 3, (0, 255, 255), -1, cv2.LINE_AA)


def draw_label(
    frame: np.ndarray,
    bbox: np.ndarray,
    lines: Iterable[str],
    color: Tuple[int, int, int],
) -> None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = np.round(bbox).astype(int)
    x1, x2 = max(0, x1), min(width - 1, x2)
    y1, y2 = max(0, y1), min(height - 1, y2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    y = max(18, y1 - 8)
    for line in reversed(list(lines)):
        (text_width, text_height), baseline = cv2.getTextSize(
            line, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1
        )
        top = max(0, y - text_height - baseline - 4)
        cv2.rectangle(frame, (x1, top), (x1 + text_width + 6, y + 2), color, -1)
        cv2.putText(
            frame,
            line,
            (x1 + 3, y - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        y = top - 2


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


class ConsoleProgress:
    """Dependency-free terminal progress for files and live streams."""

    def __init__(
        self,
        total_frames: int,
        update_interval: float = 0.5,
        width: int = 28,
    ):
        self.total_frames = max(0, int(total_frames))
        self.update_interval = max(0.1, update_interval)
        self.width = max(10, width)
        self.started = time.perf_counter()
        self.last_update = 0.0
        self.last_line_length = 0
        self.last_frame_index = -1

    def update(self, frame_index: int, force: bool = False) -> None:
        now = time.perf_counter()
        if not force and now - self.last_update < self.update_interval:
            return
        elapsed = max(1e-6, now - self.started)
        processing_fps = frame_index / elapsed
        if self.total_frames > 0:
            completed = min(1.0, frame_index / self.total_frames)
            filled = min(self.width, int(round(completed * self.width)))
            bar = "#" * filled + "-" * (self.width - filled)
            remaining_frames = max(0, self.total_frames - frame_index)
            eta = remaining_frames / processing_fps if processing_fps > 0 else 0.0
            line = (
                f"[{bar}] {completed * 100:6.2f}% "
                f"{frame_index}/{self.total_frames} "
                f"{processing_fps:5.2f} FPS "
                f"elapsed {_format_duration(elapsed)} "
                f"ETA {_format_duration(eta)}"
            )
        else:
            spinner = "|/-\\"[frame_index % 4]
            line = (
                f"[{spinner}] frames {frame_index} "
                f"{processing_fps:5.2f} FPS "
                f"elapsed {_format_duration(elapsed)}"
            )
        padding = " " * max(0, self.last_line_length - len(line))
        sys.stdout.write("\r" + line + padding)
        sys.stdout.flush()
        self.last_line_length = len(line)
        self.last_frame_index = frame_index
        self.last_update = now

    def close(self, frame_index: int) -> None:
        if frame_index != self.last_frame_index:
            self.update(frame_index, force=True)
        if self.last_line_length:
            sys.stdout.write("\n")
            sys.stdout.flush()


def main() -> int:
    args = parse_args()
    source = resolve_source(args.input)
    live_source = is_live_source(source)
    realtime_mode = live_source or bool(args.rtsp_output)
    latest_frame_mode = (
        realtime_mode if args.latest_frame is None else args.latest_frame
    )
    if args.output_fps < 0:
        raise ValueError("--output-fps must be non-negative")
    if not 0 <= args.encoder_crf <= 51:
        raise ValueError("--encoder-crf must be between 0 and 51")
    if args.candidate_hold_seconds <= 0:
        raise ValueError("--candidate-hold-seconds must be positive")
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive")
    if live_source and args.frame_stride != 1:
        raise ValueError("--frame-stride is only supported for offline files")
    device = choose_device(args.device)
    cache_dir = Path(args.cache_dir).resolve()
    torch_cache = cache_dir / "torch"
    mmengine_cache = cache_dir / "mmengine"
    torch_cache.mkdir(parents=True, exist_ok=True)
    mmengine_cache.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(torch_cache)
    os.environ["MMENGINE_HOME"] = str(mmengine_cache)
    print(f"Loading {args.pose_model!r} on {device}. First run may download model weights.")
    inferencer = MMPoseInferencer(pose2d=args.pose_model, device=device)

    capture = open_capture(source, low_latency=latest_frame_mode)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open input: {args.input}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = (
        source_total_frames if source_total_frames > 0 and not live_source else 0
    )
    if args.max_frames:
        total_frames = min(total_frames, args.max_frames) if total_frames else args.max_frames
    effective_fps = input_fps if input_fps > 1 else 25.0
    output_fps = args.output_fps if args.output_fps > 0 else effective_fps
    roi_polygon = parse_roi_polygon(args.roi_polygon, width, height)
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("Input stream did not report a valid frame size")

    direct_writers: List[FFmpegH264Writer] = []
    realtime_publishers: List[RealtimeFramePublisher] = []

    def add_output(target: str, rtsp: bool) -> None:
        writer = FFmpegH264Writer(
            target=target,
            fps=output_fps,
            size=(width, height),
            ffmpeg=args.ffmpeg,
            crf=args.encoder_crf,
            bitrate=args.encoder_bitrate,
            preset=args.encoder_preset,
            rtsp=rtsp,
        )
        if realtime_mode:
            realtime_publishers.append(RealtimeFramePublisher(writer, output_fps))
        else:
            direct_writers.append(writer)

    if args.output:
        add_output(args.output, rtsp=False)
    if args.rtsp_output:
        add_output(args.rtsp_output, rtsp=True)

    latest_reader = (
        LatestFrameReader(
            capture,
            pace_fps=effective_fps if not live_source else 0.0,
        )
        if latest_frame_mode
        else None
    )
    json_file = None
    if args.jsonl:
        json_path = Path(args.jsonl)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_file = json_path.open("w", encoding="utf-8")

    rule_config = RuleConfig(
        keypoint_threshold=args.keypoint_thr,
        holding_threshold=args.holding_thr,
        head_pitch_threshold=args.head_pitch_thr,
        head_neck_distance_max=args.head_neck_max,
    )
    quality_config = DetectionQualityConfig(
        min_detection_score=args.track_low_score,
        max_box_area_ratio=args.max_box_area_ratio,
        min_box_area_ratio=args.min_box_area_ratio,
        min_aspect_ratio=args.min_person_aspect,
        max_aspect_ratio=args.max_person_aspect,
        min_box_height=args.min_box_height,
        min_valid_keypoints=args.min_valid_keypoints,
    )
    tracker = ByteTrackLite(
        iou_threshold=args.track_iou,
        max_age=args.track_max_age,
        high_score_threshold=args.track_high_score,
        low_score_threshold=args.track_low_score,
        center_distance_threshold=args.track_center_distance,
    )
    temporal = PilotStateMachine(
        args.history,
        args.min_positive,
        exit_frames=max(1, int(round(args.candidate_hold_seconds * effective_fps))),
        exit_seconds=args.candidate_hold_seconds,
    )
    frame_index = 0
    processed_frames = 0
    dropped_frames = 0
    fps_ema = 0.0
    progress = ConsoleProgress(total_frames)

    try:
        while True:
            if latest_reader is not None:
                packet = latest_reader.read_after(frame_index)
                if packet is None:
                    if latest_reader.ended:
                        break
                    continue
                dropped_frames += max(0, packet.sequence - frame_index - 1)
            else:
                if frame_index > 0 and args.frame_stride > 1:
                    reached_end = False
                    for _ in range(args.frame_stride - 1):
                        if not capture.grab():
                            reached_end = True
                            break
                        dropped_frames += 1
                    if reached_end:
                        break
                ok, frame = capture.read()
                if not ok:
                    break
                next_index = 1 if frame_index == 0 else frame_index + args.frame_stride
                packet = CapturedFrame(next_index, time.monotonic(), frame)
            frame_index = packet.sequence
            frame = packet.image
            processed_frames += 1
            timeline_seconds = (
                packet.captured_at if live_source else frame_index / effective_fps
            )
            started = time.perf_counter()
            raw_people = run_pose(inferencer, frame)
            people = []
            rejected_quality: Dict[str, int] = {}
            for keypoints, scores, bbox, detection_score in raw_people:
                quality = evaluate_detection_quality(
                    bbox,
                    detection_score,
                    scores,
                    frame.shape,
                    quality_config,
                    roi_polygon,
                )
                if not quality.valid:
                    rejected_quality[quality.reason] = (
                        rejected_quality.get(quality.reason, 0) + 1
                    )
                    continue
                clipped_bbox = clip_bbox_xyxy(bbox, width, height)
                people.append(
                    (keypoints, scores, clipped_bbox, detection_score, quality)
                )

            track_ids = tracker.update(
                (person[2] for person in people),
                frame_index,
                scores=(person[3] for person in people),
            )
            frame_people = []

            for detection_index, (
                keypoints,
                scores,
                bbox,
                detection_score,
                detection_quality,
            ) in enumerate(people):
                # ByteTrack deliberately does not create new IDs from low-score
                # detections; those detections are only used to recover tracks.
                if detection_index not in track_ids:
                    rejected_quality["unconfirmed_low_score"] = (
                        rejected_quality.get("unconfirmed_low_score", 0) + 1
                    )
                    continue
                track_id = track_ids[detection_index]
                features = analyze_pose(
                    keypoints,
                    scores,
                    rule_config,
                    bbox,
                    image_shape=frame.shape,
                )
                frame_candidate = features.frame_candidate
                normalized_speed = tracker.normalized_speed(track_id)
                # Quality and stationarity can only modulate existing posture
                # evidence. They cannot create evidence, and a single-frame
                # head-down estimate is intentionally excluded.
                evidence_score = features.holding_score * features.pose_quality
                stationarity = max(
                    0.0, 1.0 - min(1.0, normalized_speed / 0.08)
                )
                evidence_score *= 0.90 + 0.10 * stationarity
                temporal_result = temporal.update(
                    track_id,
                    frame_candidate,
                    frame_index,
                    evidence_score=evidence_score,
                    head_down=features.head_down,
                    timestamp=timeline_seconds,
                )
                offer_controller_roi = controller_roi_candidate(
                    features,
                    temporal_result.state,
                    args.controller_candidate_mode,
                )

                if temporal_result.stable:
                    color = (0, 0, 255)
                    status = "CANDIDATE"
                elif temporal_result.state == "observing":
                    color = (0, 165, 255)
                    status = "OBSERVING"
                else:
                    color = (0, 200, 0)
                    status = "PERSON"

                head_text = (
                    "?" if features.head_down is None else str(int(features.head_down))
                )
                lines = [
                    f"P{track_id:03d} {status}",
                    (
                        f"hold={features.holding_score:.2f} head={head_text} "
                        f"grip={int(features.grip_geometry_valid)} "
                        f"conf={temporal_result.confidence:.2f}"
                    ),
                ]
                if features.head_pitch_score is not None:
                    lines.append(
                        f"pitch={features.head_pitch_score:.2f} "
                        f"switch={temporal_result.head_switches}"
                    )
                draw_label(frame, bbox, lines, color)
                if args.draw_keypoints:
                    draw_pose(frame, keypoints, scores, args.keypoint_thr)

                frame_people.append(
                    {
                        "track_id": track_id,
                        # Keep bbox for backward compatibility; bbox_xyxy is the
                        # explicit field that new ROI tooling should consume.
                        "bbox": [round(float(value), 1) for value in bbox],
                        "bbox_xyxy": [round(float(value), 2) for value in bbox],
                        "keypoints": serialize_keypoints(keypoints, scores),
                        "keypoint_layout": "halpe_body26",
                        "detection_score": round(float(detection_score), 3),
                        "detection_quality": {
                            "box_area_ratio": round(
                                detection_quality.box_area_ratio, 5
                            ),
                            "aspect_ratio": round(
                                detection_quality.aspect_ratio, 3
                            ),
                            "valid_keypoints": detection_quality.valid_keypoints,
                            "mean_keypoint_score": round(
                                detection_quality.mean_keypoint_score, 3
                            ),
                        },
                        "state": temporal_result.state,
                        "candidate_confidence": round(
                            temporal_result.confidence, 3
                        ),
                        "stable_candidate": temporal_result.stable,
                        "frame_candidate": frame_candidate,
                        "controller_roi_candidate": offer_controller_roi,
                        "candidate_level": features.candidate_level,
                        "strong_candidate": features.strong_candidate,
                        "weak_candidate": features.weak_candidate,
                        "operation_pose_type": features.operation_pose_type,
                        "holding_score": round(features.holding_score, 3),
                        "pose_quality": round(features.pose_quality, 3),
                        "shoulder_width_pixels": round(
                            features.shoulder_width_pixels, 2
                        ),
                        "person_height_ratio": (
                            None
                            if features.person_height_ratio is None
                            else round(features.person_height_ratio, 5)
                        ),
                        "scale_quality": round(features.scale_quality, 3),
                        "scale_level": features.scale_level,
                        "loose_condition_count": features.loose_condition_count,
                        "hips_available": features.hips_available,
                        "head_down": features.head_down,
                        "head_pitch_score": (
                            None
                            if features.head_pitch_score is None
                            else round(features.head_pitch_score, 3)
                        ),
                        "head_state_confidence": round(
                            features.head_state_confidence, 3
                        ),
                        "head_gap_ratio": (
                            None
                            if features.head_gap_ratio is None
                            else round(features.head_gap_ratio, 3)
                        ),
                        "wrist_distance_ratio": (
                            None
                            if features.wrist_distance_ratio is None
                            else round(features.wrist_distance_ratio, 3)
                        ),
                        "wrist_height_diff_ratio": (
                            None
                            if features.wrist_height_diff_ratio is None
                            else round(features.wrist_height_diff_ratio, 3)
                        ),
                        "wrist_torso_ratio": (
                            None
                            if features.wrist_torso_ratio is None
                            else round(features.wrist_torso_ratio, 3)
                        ),
                        "grip_geometry_valid": features.grip_geometry_valid,
                        "hands_close": features.hands_close,
                        "hands_level": features.hands_level,
                        "in_vertical_zone": features.in_vertical_zone,
                        "in_center_zone": features.in_center_zone,
                        "elbows_bent": features.elbows_bent,
                        "bbox_truncated": features.bbox_truncated,
                        "back_facing": features.back_facing,
                        "hands_behind_suspected": (
                            features.hands_behind_suspected
                        ),
                        "penalty_reason": features.penalty_reason,
                        "left_elbow_angle": (
                            None
                            if features.left_elbow_angle is None
                            else round(features.left_elbow_angle, 1)
                        ),
                        "right_elbow_angle": (
                            None
                            if features.right_elbow_angle is None
                            else round(features.right_elbow_angle, 1)
                        ),
                        "normalized_speed": round(normalized_speed, 4),
                        "head_switches": temporal_result.head_switches,
                        "vote": [
                            temporal_result.positive,
                            temporal_result.total,
                        ],
                        "reason": features.reason,
                    }
                )

            temporal.prune(
                frame_index,
                timestamp=timeline_seconds,
                max_age_seconds=max(10.0, args.candidate_hold_seconds * 2.0),
            )
            elapsed = max(1e-6, time.perf_counter() - started)
            instant_fps = 1.0 / elapsed
            fps_ema = instant_fps if fps_ema == 0 else 0.9 * fps_ema + 0.1 * instant_fps
            pipeline_age_ms = max(0.0, (time.monotonic() - packet.captured_at) * 1000.0)
            cv2.putText(
                frame,
                (
                    f"pose FPS {fps_ema:.1f} | frame {frame_index} | "
                    f"drop {dropped_frames} | age {pipeline_age_ms:.0f}ms"
                ),
                (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if json_file:
                payload = {
                    "schema_version": "1.0",
                    "source_video": str(args.input),
                    "source_type": "live" if live_source else "file",
                    "frame_index_base": 1,
                    "frame": frame_index,
                    "frame_index": frame_index,
                    "processed_frame": processed_frames,
                    "frame_stride": args.frame_stride,
                    "timestamp_ms": round(timeline_seconds * 1000.0, 1),
                    "fps": round(effective_fps, 6),
                    "image_width": width,
                    "image_height": height,
                    "dropped_frames": dropped_frames,
                    "pipeline_age_ms": round(pipeline_age_ms, 1),
                    "people": frame_people,
                    "raw_detection_count": len(raw_people),
                    "quality_rejections": rejected_quality,
                }
                json_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
            for writer in direct_writers:
                writer.write(frame)
            for publisher in realtime_publishers:
                publisher.update(frame)
            if not args.no_show:
                cv2.imshow("RTMPose candidate marker", frame)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
            progress.update(processed_frames if live_source else frame_index)
            if args.max_frames and processed_frames >= args.max_frames:
                break
    finally:
        body_failed = sys.exc_info()[0] is not None
        cleanup_errors: List[BaseException] = []
        progress.close(processed_frames if live_source else frame_index)
        if latest_reader is not None:
            latest_reader.close()
        else:
            capture.release()
        for publisher in realtime_publishers:
            try:
                publisher.release()
            except BaseException as error:
                cleanup_errors.append(error)
        for writer in direct_writers:
            try:
                writer.release()
            except BaseException as error:
                cleanup_errors.append(error)
        if json_file:
            json_file.close()
        cv2.destroyAllWindows()
        for error in cleanup_errors:
            print(f"Output cleanup error: {error}", file=sys.stderr)
        if cleanup_errors and not body_failed:
            raise cleanup_errors[0]

    print(
        f"Done. Processed {processed_frames} frames; "
        f"discarded {dropped_frames} stale frames."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
