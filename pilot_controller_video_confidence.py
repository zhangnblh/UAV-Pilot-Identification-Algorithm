"""Joint UAV-pilot analysis with per-person fused pilot confidence.

The script keeps pose evidence and controller evidence separate:

1. ``app.py`` / RTMPose detects people and Body26 keypoints.
2. ``pose_rules_2.py`` filters detections, tracks people, and scores posture.
3. Every eligible tracked person is cropped into the same square 640x640 ROI
   format used to train ``final_model2.0``.
4. RTMDet detects a controller inside each person ROI.
5. A controller is confirmed for a track when at least 3 of the latest 5
   evaluated samples exceed the controller score threshold.

Posture is a high-recall routing/prior signal.  A temporally confirmed
controller is the hard evidence used to label a person ``CONFIRMED_PILOT``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = SCRIPT_DIR / "final_model2.0"


@dataclass(frozen=True)
class RoiTransform:
    """Mapping between one square person ROI and the original video frame."""

    crop_xyxy: Tuple[float, float, float, float]
    output_size: int

    @property
    def source_side(self) -> float:
        return float(self.crop_xyxy[2] - self.crop_xyxy[0])

    def roi_box_to_frame(
        self,
        roi_box: Sequence[float],
        frame_width: int,
        frame_height: int,
    ) -> List[float]:
        scale = self.source_side / float(self.output_size)
        x1 = self.crop_xyxy[0] + float(roi_box[0]) * scale
        y1 = self.crop_xyxy[1] + float(roi_box[1]) * scale
        x2 = self.crop_xyxy[0] + float(roi_box[2]) * scale
        y2 = self.crop_xyxy[1] + float(roi_box[3]) * scale
        return [
            float(np.clip(x1, 0, frame_width)),
            float(np.clip(y1, 0, frame_height)),
            float(np.clip(x2, 0, frame_width)),
            float(np.clip(y2, 0, frame_height)),
        ]


def square_person_roi(
    frame: np.ndarray,
    bbox_xyxy: Sequence[float],
    output_size: int = 640,
    expand_x: float = 0.30,
    expand_y: float = 0.10,
    pad_value: int = 114,
) -> Tuple[np.ndarray, RoiTransform]:
    """Crop a padded square person ROI without stretching its aspect ratio."""

    x1, y1, x2, y2 = map(float, bbox_xyxy[:4])
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    center_x = 0.5 * (x1 + x2)
    center_y = 0.5 * (y1 + y2)
    side = max(width * (1.0 + expand_x), height * (1.0 + expand_y), 2.0)
    half = 0.5 * side
    crop_box = (center_x - half, center_y - half, center_x + half, center_y + half)

    ix1, iy1 = math.floor(crop_box[0]), math.floor(crop_box[1])
    ix2, iy2 = math.ceil(crop_box[2]), math.ceil(crop_box[3])
    crop_width = max(1, ix2 - ix1)
    crop_height = max(1, iy2 - iy1)
    canvas = np.full(
        (crop_height, crop_width, 3), int(pad_value), dtype=np.uint8
    )

    frame_height, frame_width = frame.shape[:2]
    sx1, sy1 = max(0, ix1), max(0, iy1)
    sx2, sy2 = min(frame_width, ix2), min(frame_height, iy2)
    if sx2 > sx1 and sy2 > sy1:
        dx1, dy1 = sx1 - ix1, sy1 - iy1
        canvas[dy1 : dy1 + sy2 - sy1, dx1 : dx1 + sx2 - sx1] = frame[
            sy1:sy2, sx1:sx2
        ]

    roi = cv2.resize(
        canvas, (output_size, output_size), interpolation=cv2.INTER_LINEAR
    )
    # Use the integer crop actually sampled into the canvas for exact mapping.
    transform = RoiTransform(
        (float(ix1), float(iy1), float(ix2), float(iy2)), output_size
    )
    return roi, transform


@dataclass(frozen=True)
class ControllerTemporalResult:
    track_id: int
    confirmed: bool
    positive: int
    total: int
    latest_score: float
    mean_positive_score: float


@dataclass
class _ControllerTrack:
    hits: Deque[bool]
    scores: Deque[float]
    last_frame: int = 0
    last_timestamp: Optional[float] = None


class ControllerTemporalConfirmer:
    """Track-level N-sample controller confirmation."""

    def __init__(
        self,
        window_size: int = 5,
        min_positive: int = 3,
        score_threshold: float = 0.20,
        require_full_window: bool = True,
    ) -> None:
        if window_size < 1:
            raise ValueError("controller window must be positive")
        if not 1 <= min_positive <= window_size:
            raise ValueError("controller min-positive must be within the window")
        if not 0.0 <= score_threshold <= 1.0:
            raise ValueError("controller score threshold must be between 0 and 1")
        self.window_size = int(window_size)
        self.min_positive = int(min_positive)
        self.score_threshold = float(score_threshold)
        self.require_full_window = bool(require_full_window)
        self._tracks: Dict[int, _ControllerTrack] = {}

    def update(
        self,
        track_id: int,
        score: float,
        frame_index: int,
        timestamp: Optional[float] = None,
    ) -> ControllerTemporalResult:
        state = self._tracks.get(track_id)
        if state is None:
            state = _ControllerTrack(
                deque(maxlen=self.window_size), deque(maxlen=self.window_size)
            )
            self._tracks[track_id] = state
        clipped = float(np.clip(score, 0.0, 1.0)) if np.isfinite(score) else 0.0
        state.hits.append(clipped >= self.score_threshold)
        state.scores.append(clipped)
        state.last_frame = int(frame_index)
        state.last_timestamp = timestamp
        return self._result(track_id, state)

    def get(self, track_id: int) -> Optional[ControllerTemporalResult]:
        state = self._tracks.get(track_id)
        return None if state is None else self._result(track_id, state)

    def _result(
        self, track_id: int, state: _ControllerTrack
    ) -> ControllerTemporalResult:
        positive = int(sum(state.hits))
        enough = (
            len(state.hits) == self.window_size
            if self.require_full_window
            else len(state.hits) >= self.min_positive
        )
        positive_scores = [
            score for score, hit in zip(state.scores, state.hits) if hit
        ]
        return ControllerTemporalResult(
            track_id=int(track_id),
            confirmed=bool(enough and positive >= self.min_positive),
            positive=positive,
            total=len(state.hits),
            latest_score=float(state.scores[-1]) if state.scores else 0.0,
            mean_positive_score=(
                float(np.mean(positive_scores)) if positive_scores else 0.0
            ),
        )

    def prune(
        self,
        frame_index: int,
        max_age_frames: int,
        timestamp: Optional[float] = None,
        max_age_seconds: Optional[float] = None,
    ) -> None:
        if timestamp is not None and max_age_seconds is not None:
            stale = [
                track_id
                for track_id, state in self._tracks.items()
                if state.last_timestamp is not None
                and timestamp - state.last_timestamp > max_age_seconds
            ]
        else:
            stale = [
                track_id
                for track_id, state in self._tracks.items()
                if frame_index - state.last_frame > max_age_frames
            ]
        for track_id in stale:
            self._tracks.pop(track_id, None)


@dataclass(frozen=True)
class ControllerDetection:
    score: float = 0.0
    bbox_roi: Optional[List[float]] = None


@dataclass(frozen=True)
class ControllerAssociation:
    accepted: bool
    reason: str
    wrist_distance_ratio: Optional[float] = None


@dataclass(frozen=True)
class ControllerEvidence:
    score: float
    level: str
    reason: str


@dataclass(frozen=True)
class PilotConfidenceResult:
    """Explainable per-track fusion output for the current frame."""

    track_id: int
    raw_score: float
    smooth_score: float
    state: str
    controller_strength: float
    controller_hit_ratio: float
    pose_score: float
    association_score: float
    controller_hits: int
    controller_samples: int
    strong_hits: int
    pose_ratio: float
    pose_mean: float
    confirm_gate: bool
    reason: str


@dataclass(frozen=True)
class _FusionControllerSample:
    normalized_score: float
    effective_hit: bool
    strong_hit: bool
    association_score: float


@dataclass
class _PilotFusionTrack:
    pose_scores: Deque[float]
    pose_broad: Deque[bool]
    controller_samples: Deque[_FusionControllerSample]
    smooth_score: float = 0.0
    initialized: bool = False
    state: str = "PERSON"
    below_release_since: Optional[float] = None
    last_controller_timestamp: Optional[float] = None
    last_frame: int = 0
    last_timestamp: float = 0.0


def controller_association_quality(
    association: ControllerAssociation,
    wrist_mid_max_ratio: float,
    wrist_max_ratio: float,
) -> float:
    """Convert an accepted geometric association into a 0..1 quality score."""

    if not association.accepted:
        return 0.0
    if association.wrist_distance_ratio is None:
        return 0.50
    limit = (
        wrist_max_ratio
        if association.reason == "near_single_wrist"
        else wrist_mid_max_ratio
    )
    return float(
        np.clip(
            1.0 - float(association.wrist_distance_ratio) / max(limit, 1e-6),
            0.0,
            1.0,
        )
    )


class PilotConfidenceFusion:
    """Track-level C/H/P/A fusion with temporal gates and release hysteresis."""

    def __init__(
        self,
        controller_window: int = 5,
        controller_min_hits: int = 3,
        controller_min_samples: int = 3,
        controller_weak_threshold: float = 0.30,
        controller_norm_upper: float = 0.70,
        controller_stale_seconds: float = 1.50,
        pose_window: int = 12,
        pose_min_valid: int = 6,
        medium_pose_ratio: float = 0.50,
        controller_strength_weight: float = 0.45,
        controller_temporal_weight: float = 0.30,
        pose_weight: float = 0.15,
        association_weight: float = 0.10,
        ema_previous_weight: float = 0.70,
        possible_threshold: float = 0.35,
        confirmed_threshold: float = 0.65,
        release_threshold: float = 0.45,
        release_seconds: float = 0.70,
    ) -> None:
        self.controller_window = int(controller_window)
        self.controller_min_hits = int(controller_min_hits)
        self.controller_min_samples = int(controller_min_samples)
        self.controller_weak_threshold = float(controller_weak_threshold)
        self.controller_norm_upper = float(controller_norm_upper)
        self.controller_stale_seconds = float(controller_stale_seconds)
        self.pose_window = int(pose_window)
        self.pose_min_valid = int(pose_min_valid)
        self.medium_pose_ratio = float(medium_pose_ratio)
        weights = np.asarray(
            [
                controller_strength_weight,
                controller_temporal_weight,
                pose_weight,
                association_weight,
            ],
            dtype=np.float64,
        )
        weight_sum = float(np.sum(weights))
        if weight_sum <= 0.0:
            raise ValueError("fusion weights must have a positive sum")
        weights /= weight_sum
        (
            self.controller_strength_weight,
            self.controller_temporal_weight,
            self.pose_weight,
            self.association_weight,
        ) = map(float, weights)
        self.ema_previous_weight = float(ema_previous_weight)
        self.possible_threshold = float(possible_threshold)
        self.confirmed_threshold = float(confirmed_threshold)
        self.release_threshold = float(release_threshold)
        self.release_seconds = float(release_seconds)
        self._tracks: Dict[int, _PilotFusionTrack] = {}

    def update(
        self,
        track_id: int,
        pose_frame_score: float,
        broad_pose_candidate: bool,
        controller_evaluated: bool,
        controller_evidence: ControllerEvidence,
        association_score: float,
        frame_index: int,
        timestamp: float,
    ) -> PilotConfidenceResult:
        state = self._tracks.get(track_id)
        if state is None:
            state = _PilotFusionTrack(
                pose_scores=deque(maxlen=self.pose_window),
                pose_broad=deque(maxlen=self.pose_window),
                controller_samples=deque(maxlen=self.controller_window),
            )
            self._tracks[track_id] = state

        pose_value = (
            float(np.clip(pose_frame_score, 0.0, 1.0))
            if np.isfinite(pose_frame_score)
            else 0.0
        )
        state.pose_scores.append(pose_value)
        state.pose_broad.append(bool(broad_pose_candidate))

        if controller_evaluated:
            if (
                state.last_controller_timestamp is not None
                and timestamp - state.last_controller_timestamp
                > self.controller_stale_seconds
            ):
                state.controller_samples.clear()
            hit = bool(controller_evidence.score > 0.0)
            strong = bool(hit and controller_evidence.level == "strong")
            normalized = 0.0
            if hit:
                normalized = float(
                    np.clip(
                        (
                            controller_evidence.score
                            - self.controller_weak_threshold
                        )
                        / max(
                            self.controller_norm_upper
                            - self.controller_weak_threshold,
                            1e-6,
                        ),
                        0.0,
                        1.0,
                    )
                )
            state.controller_samples.append(
                _FusionControllerSample(
                    normalized_score=normalized,
                    effective_hit=hit,
                    strong_hit=strong,
                    association_score=(
                        float(np.clip(association_score, 0.0, 1.0))
                        if hit
                        else 0.0
                    ),
                )
            )
            state.last_controller_timestamp = float(timestamp)

        state.last_frame = int(frame_index)
        state.last_timestamp = float(timestamp)
        return self._calculate(track_id, state, float(timestamp))

    def _calculate(
        self,
        track_id: int,
        state: _PilotFusionTrack,
        timestamp: float,
    ) -> PilotConfidenceResult:
        controller_fresh = bool(
            state.last_controller_timestamp is not None
            and timestamp - state.last_controller_timestamp
            <= self.controller_stale_seconds
        )
        samples = list(state.controller_samples) if controller_fresh else []
        sample_count = len(samples)
        hits = [sample for sample in samples if sample.effective_hit]
        hit_count = len(hits)
        strong_count = sum(sample.strong_hit for sample in samples)
        hit_ratio = hit_count / sample_count if sample_count else 0.0

        top_scores = sorted(
            (sample.normalized_score for sample in hits), reverse=True
        )[:3]
        controller_strength = (
            float(np.mean(top_scores)) if top_scores else 0.0
        )
        association_score = (
            float(np.mean([sample.association_score for sample in hits]))
            if hits
            else 0.0
        )

        pose_scores = list(state.pose_scores)
        pose_broad = list(state.pose_broad)
        pose_count = len(pose_scores)
        pose_mean = float(np.mean(pose_scores)) if pose_scores else 0.0
        pose_ratio = (
            float(sum(pose_broad)) / pose_count if pose_count else 0.0
        )
        pose_reliability = min(
            pose_count / max(float(self.pose_min_valid), 1.0), 1.0
        )
        pose_score = float(
            np.clip(
                (0.60 * pose_mean + 0.40 * pose_ratio)
                * pose_reliability,
                0.0,
                1.0,
            )
        )

        raw_score = (
            self.controller_strength_weight * controller_strength
            + self.controller_temporal_weight * hit_ratio
            + self.pose_weight * pose_score
            + self.association_weight * association_score
        )
        raw_score = float(np.clip(raw_score, 0.0, 1.0))

        enough_samples = sample_count >= self.controller_min_samples
        temporal_gate = bool(
            enough_samples and hit_count >= self.controller_min_hits
        )
        strong_gate = bool(
            temporal_gate and strong_count >= self.controller_min_hits
        )
        medium_gate = bool(
            temporal_gate and pose_ratio >= self.medium_pose_ratio
        )
        confirm_gate = bool(strong_gate or medium_gate)

        if hit_count == 0:
            # A stable broad pose remains visible as POSSIBLE, but the missing
            # controller gate makes CONFIRMED impossible.
            raw_score = min(
                raw_score, self.confirmed_threshold - 1e-6
            )
            if (
                pose_count >= self.pose_min_valid
                and pose_ratio >= self.medium_pose_ratio
            ):
                raw_score = max(raw_score, self.possible_threshold)
        elif hit_count == 1:
            raw_score = min(raw_score, 0.49)
        if strong_gate:
            raw_score = max(raw_score, 0.70)
        elif medium_gate:
            raw_score = max(raw_score, 0.65)

        if state.initialized:
            smooth_score = (
                self.ema_previous_weight * state.smooth_score
                + (1.0 - self.ema_previous_weight) * raw_score
            )
        else:
            smooth_score = raw_score
            state.initialized = True

        if confirm_gate and raw_score >= self.confirmed_threshold:
            smooth_score = max(smooth_score, self.confirmed_threshold)
        smooth_score = float(np.clip(smooth_score, 0.0, 1.0))

        if state.state == "CONFIRMED_PILOT":
            if confirm_gate and smooth_score >= self.confirmed_threshold:
                next_state = "CONFIRMED_PILOT"
                state.below_release_since = None
                reason = (
                    "strong_controller_temporal"
                    if strong_gate
                    else "medium_controller_temporal_with_broad_pose"
                )
            elif smooth_score >= self.release_threshold:
                next_state = "CONFIRMED_PILOT"
                state.below_release_since = None
                reason = "confirmed_score_hysteresis"
            else:
                if state.below_release_since is None:
                    state.below_release_since = timestamp
                release_elapsed = timestamp - state.below_release_since
                if release_elapsed < self.release_seconds:
                    next_state = "CONFIRMED_PILOT"
                    reason = "confirmed_time_hysteresis"
                elif smooth_score >= self.possible_threshold:
                    next_state = "POSSIBLE_PILOT"
                    state.below_release_since = None
                    reason = "released_to_possible"
                else:
                    next_state = "PERSON"
                    state.below_release_since = None
                    reason = "released_to_person"
        elif confirm_gate and smooth_score >= self.confirmed_threshold:
            next_state = "CONFIRMED_PILOT"
            state.below_release_since = None
            reason = (
                "strong_controller_temporal"
                if strong_gate
                else "medium_controller_temporal_with_broad_pose"
            )
        elif smooth_score >= self.possible_threshold:
            next_state = "POSSIBLE_PILOT"
            state.below_release_since = None
            reason = (
                "controller_pending"
                if hit_count > 0
                else "pose_only_candidate"
            )
        else:
            next_state = "PERSON"
            state.below_release_since = None
            reason = (
                "controller_evidence_stale"
                if (
                    not controller_fresh
                    and state.last_controller_timestamp is not None
                )
                else "insufficient_evidence"
            )

        state.smooth_score = smooth_score
        state.state = next_state
        return PilotConfidenceResult(
            track_id=int(track_id),
            raw_score=raw_score,
            smooth_score=smooth_score,
            state=next_state,
            controller_strength=controller_strength,
            controller_hit_ratio=hit_ratio,
            pose_score=pose_score,
            association_score=association_score,
            controller_hits=hit_count,
            controller_samples=sample_count,
            strong_hits=int(strong_count),
            pose_ratio=pose_ratio,
            pose_mean=pose_mean,
            confirm_gate=confirm_gate,
            reason=reason,
        )

    def prune(
        self,
        frame_index: int,
        max_age_frames: int,
        timestamp: Optional[float] = None,
        max_age_seconds: Optional[float] = None,
    ) -> None:
        if timestamp is not None and max_age_seconds is not None:
            stale = [
                track_id
                for track_id, state in self._tracks.items()
                if timestamp - state.last_timestamp > max_age_seconds
            ]
        else:
            stale = [
                track_id
                for track_id, state in self._tracks.items()
                if frame_index - state.last_frame > max_age_frames
            ]
        for track_id in stale:
            self._tracks.pop(track_id, None)


def associate_controller_to_person(
    controller_box: Optional[Sequence[float]],
    person_box: Sequence[float],
    keypoints: np.ndarray,
    keypoint_scores: np.ndarray,
    keypoint_threshold: float = 0.35,
    wrist_mid_max_ratio: float = 0.22,
    wrist_max_ratio: float = 0.18,
    person_expand_ratio: float = 0.10,
    person_y_max_ratio: float = 0.85,
) -> ControllerAssociation:
    """Verify that a controller belongs to this person, not a nearby ROI."""
    if controller_box is None:
        return ControllerAssociation(False, "no_controller_box")
    box = np.asarray(controller_box, dtype=np.float32).reshape(-1)
    person = np.asarray(person_box, dtype=np.float32).reshape(-1)
    if len(box) < 4 or len(person) < 4:
        return ControllerAssociation(False, "invalid_box")
    if not np.all(np.isfinite(box[:4])) or not np.all(np.isfinite(person[:4])):
        return ControllerAssociation(False, "invalid_box")
    px1, py1, px2, py2 = map(float, person[:4])
    person_width = max(1.0, px2 - px1)
    person_height = max(1.0, py2 - py1)
    controller_center = np.asarray(
        [0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])], dtype=np.float32
    )
    inside_person_zone = bool(
        px1 - person_expand_ratio * person_width <= controller_center[0]
        <= px2 + person_expand_ratio * person_width
        and py1 - 0.05 * person_height <= controller_center[1]
        <= py1 + person_y_max_ratio * person_height
    )
    if not inside_person_zone:
        return ControllerAssociation(False, "outside_person_zone")
    points = np.asarray(keypoints, dtype=np.float32)
    scores = np.asarray(keypoint_scores, dtype=np.float32).reshape(-1)
    valid_wrists = [
        i for i in (9, 10)
        if points.ndim == 2 and points.shape[1] >= 2 and i < len(points)
        and i < len(scores) and np.all(np.isfinite(points[i, :2]))
        and np.isfinite(scores[i]) and scores[i] >= keypoint_threshold
    ]
    if len(valid_wrists) == 2:
        wrists = points[valid_wrists, :2]
        midpoint_distance = float(
            np.linalg.norm(controller_center - np.mean(wrists, axis=0)) / person_height
        )
        nearest_distance = float(
            min(np.linalg.norm(controller_center - wrist) for wrist in wrists)
            / person_height
        )
        accepted = midpoint_distance <= wrist_mid_max_ratio or nearest_distance <= wrist_max_ratio
        return ControllerAssociation(
            bool(accepted), "near_wrists" if accepted else "far_from_wrists",
            min(midpoint_distance, nearest_distance)
        )
    if len(valid_wrists) == 1:
        distance = float(
            np.linalg.norm(controller_center - points[valid_wrists[0], :2]) / person_height
        )
        accepted = distance <= wrist_max_ratio
        return ControllerAssociation(
            bool(accepted), "near_single_wrist" if accepted else "far_from_single_wrist", distance
        )
    return ControllerAssociation(True, "person_zone_fallback")


def classify_controller_evidence(
    raw_score: float,
    spatially_associated: bool,
    broad_pose_candidate: bool,
    weak_threshold: float = 0.30,
    strong_threshold: float = 0.50,
) -> ControllerEvidence:
    """Apply strong/weak/negative controller fusion policy."""
    score = float(np.clip(raw_score, 0.0, 1.0)) if np.isfinite(raw_score) else 0.0
    if not spatially_associated:
        return ControllerEvidence(0.0, "negative", "spatial_mismatch")
    if score >= strong_threshold:
        return ControllerEvidence(score, "strong", "strong_controller")
    if score >= weak_threshold and broad_pose_candidate:
        return ControllerEvidence(score, "broad", "controller_plus_broad_pose")
    if score >= weak_threshold:
        return ControllerEvidence(0.0, "negative", "broad_pose_required")
    return ControllerEvidence(0.0, "negative", "below_controller_threshold")


@dataclass
class PersonRecord:
    track_id: int
    keypoints: np.ndarray
    keypoint_scores: np.ndarray
    bbox: np.ndarray
    detection_score: float
    detection_quality: Any
    pose_features: Any
    pose_temporal: Any
    offer_controller_roi: bool
    pose_evidence_score: float
    controller_evaluated: bool = False
    roi_transform: Optional[RoiTransform] = None
    controller_detection: ControllerDetection = field(
        default_factory=ControllerDetection
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RTMPose + person ROI RTMDet + 5-sample UAV-pilot analysis."
    )
    parser.add_argument(
        "--input", required=True, help="Video path, camera index, or RTSP/HTTP URL."
    )
    parser.add_argument("--output", help="Optional annotated H.264 MP4 path.")
    parser.add_argument("--jsonl", help="Optional detailed per-frame JSONL output.")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Offline video only: process every Nth source frame.",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda:0, etc.")
    parser.add_argument(
        "--camera-width",
        type=int,
        default=0,
        help="USB camera requested width; 0 keeps the camera default.",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=0,
        help="USB camera requested height; 0 keeps the camera default.",
    )
    parser.add_argument(
        "--camera-fps",
        type=float,
        default=0.0,
        help="USB camera requested capture FPS; 0 keeps the camera default.",
    )
    parser.add_argument(
        "--output-fps",
        type=float,
        default=0.0,
        help=(
            "Saved-video FPS. For live input, 0 uses 5 FPS; set this close "
            "to the displayed inference FPS to avoid fast/slow playback."
        ),
    )
    parser.add_argument("--pose-model", default="body26")
    parser.add_argument(
        "--cache-dir",
        default=str(SCRIPT_DIR / ".cache"),
        help="RTMPose/MMEngine model cache (default: PROJECT/.cache).",
    )

    parser.add_argument(
        "--controller-config",
        default=str(DEFAULT_MODEL_DIR / "handheld_rtmdet_tiny_960.py"),
    )
    parser.add_argument(
        "--controller-checkpoint",
        default=str(DEFAULT_MODEL_DIR / "best_model.pth"),
    )
    parser.add_argument(
        "--controller-score-thr", type=float, default=0.30,
        help="Weak controller threshold; requires broad pose evidence.",
    )
    parser.add_argument(
        "--controller-strong-thr", type=float, default=0.50,
        help="Strong controller threshold; pose evidence is not required.",
    )
    parser.add_argument("--controller-draw-thr", type=float, default=0.30)
    parser.add_argument("--controller-window", type=int, default=5)
    parser.add_argument("--controller-min-positive", type=int, default=3)
    parser.add_argument(
        "--controller-every",
        type=int,
        default=1,
        help="Run controller inference every N processed pose frames.",
    )
    # GTX 1060 6 GB shares memory with RTMPose; keep the safe default small.
    parser.add_argument("--controller-batch-size", type=int, default=1)
    parser.add_argument(
        "--controller-candidate-mode",
        choices=("all", "broad", "strict"),
        default="all",
        help="all is safest for recall; broad/strict save controller GPU work.",
    )
    parser.add_argument("--roi-output-size", type=int, default=640)
    parser.add_argument("--person-expand-x", type=float, default=0.30)
    parser.add_argument("--person-expand-y", type=float, default=0.10)
    parser.add_argument("--controller-wrist-mid-max", type=float, default=0.22)
    parser.add_argument("--controller-wrist-max", type=float, default=0.18)
    parser.add_argument("--controller-person-expand", type=float, default=0.10)
    parser.add_argument(
        "--min-roi-source-side",
        type=float,
        default=32.0,
        help="Skip controller inference below this pre-resize square side.",
    )
    parser.add_argument("--pilot-hold-seconds", type=float, default=1.5)
    parser.add_argument(
        "--fusion-controller-upper",
        type=float,
        default=0.70,
        help="Raw controller score mapped to normalized fusion strength 1.0.",
    )
    parser.add_argument("--fusion-controller-min-samples", type=int, default=3)
    parser.add_argument("--fusion-controller-stale-seconds", type=float, default=1.50)
    parser.add_argument("--fusion-pose-window", type=int, default=12)
    parser.add_argument("--fusion-pose-min-valid", type=int, default=6)
    parser.add_argument("--fusion-medium-pose-ratio", type=float, default=0.50)
    parser.add_argument("--fusion-controller-weight", type=float, default=0.45)
    parser.add_argument("--fusion-temporal-weight", type=float, default=0.30)
    parser.add_argument("--fusion-pose-weight", type=float, default=0.15)
    parser.add_argument("--fusion-association-weight", type=float, default=0.10)
    parser.add_argument("--fusion-ema-previous", type=float, default=0.70)
    parser.add_argument("--fusion-possible-thr", type=float, default=0.35)
    parser.add_argument("--fusion-confirmed-thr", type=float, default=0.65)
    parser.add_argument("--fusion-release-thr", type=float, default=0.45)
    parser.add_argument("--fusion-release-seconds", type=float, default=0.70)

    parser.add_argument("--keypoint-thr", type=float, default=0.35)
    parser.add_argument("--holding-thr", type=float, default=0.80)
    parser.add_argument("--head-pitch-thr", type=float, default=0.42)
    parser.add_argument("--head-neck-max", type=float, default=0.90)
    parser.add_argument("--pose-history", type=int, default=12)
    parser.add_argument("--pose-min-positive", type=int, default=8)
    parser.add_argument("--candidate-hold-seconds", type=float, default=5.0)
    parser.add_argument(
        "--draw-keypoints",
        action="store_true",
        help=(
            "Draw pose keypoints for every tracked person. Confirmed pilots "
            "are drawn automatically even when this option is omitted."
        ),
    )

    parser.add_argument("--track-iou", type=float, default=0.15)
    parser.add_argument("--track-center-distance", type=float, default=0.80)
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
        help="Optional area: x1,y1;x2,y2;... in pixels or normalized 0..1.",
    )

    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--encoder-crf", type=int, default=20)
    parser.add_argument("--encoder-bitrate", default="3M")
    parser.add_argument("--encoder-preset", default="veryfast")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.frame_stride < 1 or args.controller_every < 1:
        raise ValueError("frame/controller stride must be positive")
    if (
        args.camera_width < 0
        or args.camera_height < 0
        or args.camera_fps < 0
        or args.output_fps < 0
    ):
        raise ValueError("camera dimensions/FPS and output FPS must be non-negative")
    if args.controller_batch_size < 1:
        raise ValueError("controller batch size must be positive")
    if args.roi_output_size < 32:
        raise ValueError("ROI output size must be at least 32")
    if args.min_roi_source_side < 1:
        raise ValueError("minimum ROI source side must be positive")
    if not 0.0 <= args.controller_score_thr <= 1.0:
        raise ValueError("controller score threshold must be between 0 and 1")
    if not args.controller_score_thr <= args.controller_strong_thr <= 1.0:
        raise ValueError(
            "controller strong threshold must be between the weak threshold and 1"
        )
    if not 0.0 <= args.controller_draw_thr <= 1.0:
        raise ValueError("controller draw threshold must be between 0 and 1")
    if args.controller_wrist_mid_max <= 0 or args.controller_wrist_max <= 0:
        raise ValueError("controller wrist-distance ratios must be positive")
    if args.controller_person_expand < 0:
        raise ValueError("controller person expansion must be non-negative")
    if args.pilot_hold_seconds < 0:
        raise ValueError("pilot hold seconds must be non-negative")
    if not (
        args.controller_score_thr
        < args.fusion_controller_upper
        <= 1.0
    ):
        raise ValueError(
            "fusion controller upper must be above weak threshold and <= 1"
        )
    if not 1 <= args.fusion_controller_min_samples <= args.controller_window:
        raise ValueError(
            "fusion controller min samples must be within controller window"
        )
    if not 1 <= args.controller_min_positive <= args.controller_window:
        raise ValueError(
            "controller min positive must be within controller window"
        )
    if args.fusion_controller_stale_seconds <= 0:
        raise ValueError("fusion controller stale seconds must be positive")
    if args.fusion_pose_window < 1 or args.fusion_pose_min_valid < 1:
        raise ValueError("fusion pose window/min valid must be positive")
    if not 0.0 <= args.fusion_medium_pose_ratio <= 1.0:
        raise ValueError("fusion medium pose ratio must be between 0 and 1")
    fusion_weights = (
        args.fusion_controller_weight,
        args.fusion_temporal_weight,
        args.fusion_pose_weight,
        args.fusion_association_weight,
    )
    if any(weight < 0 for weight in fusion_weights) or sum(fusion_weights) <= 0:
        raise ValueError("fusion weights must be non-negative with positive sum")
    if not 0.0 <= args.fusion_ema_previous < 1.0:
        raise ValueError("fusion EMA previous weight must be in [0, 1)")
    if not (
        0.0
        <= args.fusion_possible_thr
        <= args.fusion_release_thr
        < args.fusion_confirmed_thr
        <= 1.0
    ):
        raise ValueError(
            "fusion thresholds must satisfy possible <= release < confirmed"
        )
    if args.fusion_release_seconds < 0:
        raise ValueError("fusion release seconds must be non-negative")
    if args.candidate_hold_seconds <= 0:
        raise ValueError("candidate hold seconds must be positive")


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def extract_best_controller(result: Any) -> ControllerDetection:
    """Read MMDetection 3.x DetDataSample (and common dict variants)."""

    instances = getattr(result, "pred_instances", None)
    if instances is None and isinstance(result, dict):
        instances = result.get("pred_instances", result.get("predictions"))
    if instances is None:
        return ControllerDetection()

    if isinstance(instances, dict):
        raw_boxes = instances.get("bboxes", [])
        raw_scores = instances.get("scores", [])
        raw_labels = instances.get("labels")
    else:
        raw_boxes = getattr(instances, "bboxes", [])
        raw_scores = getattr(instances, "scores", [])
        raw_labels = getattr(instances, "labels", None)

    boxes = _numpy(raw_boxes).reshape(-1, 4)
    scores = _numpy(raw_scores).reshape(-1)
    labels = (
        np.zeros(len(scores), dtype=np.int64)
        if raw_labels is None
        else _numpy(raw_labels).reshape(-1).astype(np.int64)
    )
    count = min(len(boxes), len(scores), len(labels))
    candidates = [
        index
        for index in range(count)
        if labels[index] == 0 and np.isfinite(scores[index])
    ]
    if not candidates:
        return ControllerDetection()
    best = max(candidates, key=lambda index: float(scores[index]))
    return ControllerDetection(
        score=float(np.clip(scores[best], 0.0, 1.0)),
        bbox_roi=[float(value) for value in boxes[best]],
    )


def run_controller_batches(
    model: Any,
    inference_detector: Any,
    rois: Sequence[np.ndarray],
    batch_size: int,
) -> List[ControllerDetection]:
    detections: List[ControllerDetection] = []
    for start in range(0, len(rois), batch_size):
        batch = list(rois[start : start + batch_size])
        raw_results = inference_detector(model, batch)
        if not isinstance(raw_results, (list, tuple)):
            raw_results = [raw_results]
        if len(raw_results) != len(batch):
            raise RuntimeError(
                "MMDetection returned an unexpected number of ROI results: "
                f"{len(raw_results)} for {len(batch)} inputs"
            )
        detections.extend(extract_best_controller(item) for item in raw_results)
    return detections


def rounded_box(box: Optional[Sequence[float]]) -> Optional[List[float]]:
    if box is None:
        return None
    return [round(float(value), 2) for value in box]


def draw_controller_box(
    frame: np.ndarray, box: Sequence[float], score: float
) -> None:
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 80, 255), 2)
    cv2.putText(
        frame,
        f"controller {score:.2f}",
        (x1, max(18, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 80, 255),
        2,
        cv2.LINE_AA,
    )


def main() -> int:
    args = parse_args()
    validate_args(args)

    # Heavy OpenMMLab imports stay inside main so geometry/temporal unit tests
    # can run even in a lightweight environment.
    try:
        import app as pose_app
        from pose_rules_2 import (
            ByteTrackLite,
            DetectionQualityConfig,
            PilotStateMachine,
            RuleConfig,
            analyze_pose,
            evaluate_detection_quality,
        )
        from mmdet.apis import inference_detector, init_detector
        from mmdet.utils import register_all_modules
    except ImportError as error:
        raise RuntimeError(
            "A unified OpenMMLab environment is required. It must provide "
            "mmpose, mmdet, mmengine, mmcv, torch, OpenCV and NumPy. "
            f"Original import error: {error}"
        ) from error

    source = pose_app.resolve_source(args.input)
    live_source = pose_app.is_live_source(source)
    if live_source and args.frame_stride != 1:
        raise ValueError("--frame-stride is only supported for offline files")
    device = pose_app.choose_device(args.device)

    config_path = Path(args.controller_config).resolve()
    checkpoint_path = Path(args.controller_checkpoint).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Controller config not found: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Controller checkpoint not found: {checkpoint_path}")

    cache_dir = Path(args.cache_dir).resolve()
    (cache_dir / "torch").mkdir(parents=True, exist_ok=True)
    (cache_dir / "mmengine").mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(cache_dir / "torch")
    os.environ["MMENGINE_HOME"] = str(cache_dir / "mmengine")

    # Resolve the mmdet:: base-config scope before parsing the RTMDet config.
    register_all_modules(init_default_scope=True)
    print(f"Loading RTMPose {args.pose_model!r} on {device} ...")
    pose_inferencer = pose_app.MMPoseInferencer(
        pose2d=args.pose_model, device=device
    )
    print(f"Loading RTMDet controller model on {device} ...")
    controller_model = init_detector(
        str(config_path), str(checkpoint_path), device=device
    )

    capture = pose_app.open_capture(source, low_latency=live_source)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open input: {args.input}")
    if isinstance(source, int):
        if args.camera_width > 0:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
        if args.camera_height > 0:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
        if args.camera_fps > 0:
            capture.set(cv2.CAP_PROP_FPS, args.camera_fps)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = float(capture.get(cv2.CAP_PROP_FPS))
    source_total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = 0 if live_source else max(0, source_total_frames)
    effective_fps = input_fps if input_fps > 1 else 25.0
    output_fps = (
        args.output_fps
        if args.output_fps > 0
        else (5.0 if live_source else effective_fps / args.frame_stride)
    )
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("Input did not report a valid frame size")
    if args.max_frames:
        total_frames = (
            min(total_frames, args.max_frames)
            if total_frames
            else args.max_frames
        )
    print(
        f"Input mode: {'LIVE latest-frame' if live_source else 'offline'} | "
        f"{width}x{height} @ {effective_fps:.2f} FPS | "
        f"saved output {output_fps:.2f} FPS"
    )

    roi_polygon = pose_app.parse_roi_polygon(args.roi_polygon, width, height)
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
    pose_temporal = PilotStateMachine(
        args.pose_history,
        args.pose_min_positive,
        exit_frames=max(1, int(round(args.candidate_hold_seconds * effective_fps))),
        exit_seconds=args.candidate_hold_seconds,
    )
    controller_temporal = ControllerTemporalConfirmer(
        window_size=args.controller_window,
        min_positive=args.controller_min_positive,
        score_threshold=args.controller_score_thr,
        require_full_window=True,
    )
    pilot_fusion = PilotConfidenceFusion(
        controller_window=args.controller_window,
        controller_min_hits=args.controller_min_positive,
        controller_min_samples=args.fusion_controller_min_samples,
        controller_weak_threshold=args.controller_score_thr,
        controller_norm_upper=args.fusion_controller_upper,
        controller_stale_seconds=args.fusion_controller_stale_seconds,
        pose_window=args.fusion_pose_window,
        pose_min_valid=args.fusion_pose_min_valid,
        medium_pose_ratio=args.fusion_medium_pose_ratio,
        controller_strength_weight=args.fusion_controller_weight,
        controller_temporal_weight=args.fusion_temporal_weight,
        pose_weight=args.fusion_pose_weight,
        association_weight=args.fusion_association_weight,
        ema_previous_weight=args.fusion_ema_previous,
        possible_threshold=args.fusion_possible_thr,
        confirmed_threshold=args.fusion_confirmed_thr,
        release_threshold=args.fusion_release_thr,
        release_seconds=args.fusion_release_seconds,
    )

    writer = None
    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg_available = bool(
            shutil.which(args.ffmpeg) or Path(args.ffmpeg).is_file()
        )
        if ffmpeg_available:
            writer = pose_app.FFmpegH264Writer(
                target=str(output_path),
                fps=output_fps,
                size=(width, height),
                ffmpeg=args.ffmpeg,
                crf=args.encoder_crf,
                bitrate=args.encoder_bitrate,
                preset=args.encoder_preset,
                rtsp=False,
            )
        else:
            print(
                "FFmpeg was not found; falling back to OpenCV MP4V output. "
                "Use --ffmpeg PATH_TO_FFMPEG.EXE for H.264 output."
            )
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                output_fps,
                (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError(
                    "Neither FFmpeg nor the OpenCV MP4V writer could open: "
                    f"{output_path}"
                )
    json_file = None
    if args.jsonl:
        json_path = Path(args.jsonl).resolve()
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_file = json_path.open("w", encoding="utf-8")

    progress = pose_app.ConsoleProgress(total_frames)
    frame_index = 0
    processed_frames = 0
    fps_ema = 0.0
    dropped_frames = 0
    last_live_sequence = 0
    live_time_origin: Optional[float] = None
    live_reader = (
        pose_app.LatestFrameReader(capture)
        if live_source
        else None
    )
    last_confirmed_at: Dict[int, float] = {}
    last_seen_at: Dict[int, float] = {}

    try:
        while True:
            if args.max_frames and processed_frames >= args.max_frames:
                break
            if live_reader is not None:
                packet = live_reader.read_after(
                    last_live_sequence, timeout=5.0
                )
                if packet is None:
                    if live_reader.ended:
                        break
                    continue
                dropped_frames += max(
                    0, packet.sequence - last_live_sequence - 1
                )
                last_live_sequence = packet.sequence
                frame_index = packet.sequence
                decoded = packet.image
                if live_time_origin is None:
                    live_time_origin = packet.captured_at
                timeline_seconds = (
                    packet.captured_at - live_time_origin
                )
            else:
                if frame_index > 0 and args.frame_stride > 1:
                    ended = False
                    for _ in range(args.frame_stride - 1):
                        if not capture.grab():
                            ended = True
                            break
                    if ended:
                        break
                ok, decoded = capture.read()
                if not ok:
                    break
                frame_index = (
                    1
                    if frame_index == 0
                    else frame_index + args.frame_stride
                )
                timeline_seconds = frame_index / effective_fps
            processed_frames += 1

            started = time.perf_counter()
            # Never crop an ROI from a frame that already contains overlays.
            raw_frame = decoded
            display_frame = decoded.copy()

            raw_people = pose_app.run_pose(pose_inferencer, raw_frame)
            people = []
            rejection_counts: Dict[str, int] = {}
            for keypoints, scores, bbox, detection_score in raw_people:
                quality = evaluate_detection_quality(
                    bbox,
                    detection_score,
                    scores,
                    raw_frame.shape,
                    quality_config,
                    roi_polygon,
                )
                if not quality.valid:
                    rejection_counts[quality.reason] = (
                        rejection_counts.get(quality.reason, 0) + 1
                    )
                    continue
                clipped_bbox = pose_app.clip_bbox_xyxy(bbox, width, height)
                people.append(
                    (keypoints, scores, clipped_bbox, detection_score, quality)
                )

            track_ids = tracker.update(
                (person[2] for person in people),
                frame_index,
                scores=(person[3] for person in people),
            )
            records: List[PersonRecord] = []
            controller_rois: List[np.ndarray] = []
            controller_record_indexes: List[int] = []
            run_controller_now = (
                (processed_frames - 1) % args.controller_every == 0
            )

            for detection_index, person in enumerate(people):
                if detection_index not in track_ids:
                    rejection_counts["unconfirmed_low_score"] = (
                        rejection_counts.get("unconfirmed_low_score", 0) + 1
                    )
                    continue
                keypoints, scores, bbox, detection_score, quality = person
                track_id = track_ids[detection_index]
                features = analyze_pose(
                    keypoints,
                    scores,
                    rule_config,
                    bbox,
                    image_shape=raw_frame.shape,
                )
                normalized_speed = tracker.normalized_speed(track_id)
                evidence_score = features.holding_score * features.pose_quality
                stationarity = max(
                    0.0, 1.0 - min(1.0, normalized_speed / 0.08)
                )
                evidence_score *= 0.90 + 0.10 * stationarity
                pose_result = pose_temporal.update(
                    track_id,
                    features.frame_candidate,
                    frame_index,
                    evidence_score=evidence_score,
                    head_down=features.head_down,
                    timestamp=timeline_seconds,
                )
                offer_roi = pose_app.controller_roi_candidate(
                    features, pose_result.state, args.controller_candidate_mode
                )
                record = PersonRecord(
                    track_id=track_id,
                    keypoints=keypoints,
                    keypoint_scores=scores,
                    bbox=bbox,
                    detection_score=float(detection_score),
                    detection_quality=quality,
                    pose_features=features,
                    pose_temporal=pose_result,
                    offer_controller_roi=offer_roi,
                    pose_evidence_score=float(evidence_score),
                )
                records.append(record)
                last_seen_at[track_id] = timeline_seconds

                if offer_roi and run_controller_now:
                    roi, transform = square_person_roi(
                        raw_frame,
                        bbox,
                        output_size=args.roi_output_size,
                        expand_x=args.person_expand_x,
                        expand_y=args.person_expand_y,
                    )
                    record.roi_transform = transform
                    if transform.source_side >= args.min_roi_source_side:
                        record.controller_evaluated = True
                        controller_record_indexes.append(len(records) - 1)
                        controller_rois.append(roi)

            if controller_rois:
                controller_detections = run_controller_batches(
                    controller_model,
                    inference_detector,
                    controller_rois,
                    args.controller_batch_size,
                )
                for record_index, detection in zip(
                    controller_record_indexes, controller_detections
                ):
                    records[record_index].controller_detection = detection

            people_payload = []
            confirmed_count = 0
            possible_count = 0
            for record in records:
                detection = record.controller_detection
                global_controller_box = None
                association = ControllerAssociation(False, "not_evaluated")
                broad_pose_candidate = pose_app.controller_roi_candidate(
                    record.pose_features, record.pose_temporal.state, "broad"
                )
                evidence = ControllerEvidence(0.0, "negative", "not_evaluated")
                if (
                    record.controller_evaluated
                    and detection.bbox_roi is not None
                    and record.roi_transform is not None
                ):
                    global_controller_box = record.roi_transform.roi_box_to_frame(
                        detection.bbox_roi, width, height
                    )
                    association = associate_controller_to_person(
                        global_controller_box,
                        record.bbox,
                        record.keypoints,
                        record.keypoint_scores,
                        keypoint_threshold=args.keypoint_thr,
                        wrist_mid_max_ratio=args.controller_wrist_mid_max,
                        wrist_max_ratio=args.controller_wrist_max,
                        person_expand_ratio=args.controller_person_expand,
                    )
                    evidence = classify_controller_evidence(
                        detection.score,
                        association.accepted,
                        broad_pose_candidate,
                        weak_threshold=args.controller_score_thr,
                        strong_threshold=args.controller_strong_thr,
                    )
                if record.controller_evaluated:
                    controller_result = controller_temporal.update(
                        record.track_id,
                        evidence.score,
                        frame_index,
                        timestamp=timeline_seconds,
                    )
                    if controller_result.confirmed:
                        last_confirmed_at[record.track_id] = timeline_seconds
                else:
                    controller_result = controller_temporal.get(record.track_id)

                confirmed_now = bool(
                    controller_result is not None and controller_result.confirmed
                )
                confirmed_held = bool(
                    record.track_id in last_confirmed_at
                    and timeline_seconds - last_confirmed_at[record.track_id]
                    <= args.pilot_hold_seconds
                )
                controller_confirmed = confirmed_now or confirmed_held

                association_score = controller_association_quality(
                    association,
                    wrist_mid_max_ratio=args.controller_wrist_mid_max,
                    wrist_max_ratio=args.controller_wrist_max,
                )
                fusion_result = pilot_fusion.update(
                    track_id=record.track_id,
                    pose_frame_score=record.pose_evidence_score,
                    broad_pose_candidate=broad_pose_candidate,
                    controller_evaluated=record.controller_evaluated,
                    controller_evidence=evidence,
                    association_score=association_score,
                    frame_index=frame_index,
                    timestamp=timeline_seconds,
                )
                final_state = fusion_result.state

                if final_state == "CONFIRMED_PILOT":
                    color = (0, 0, 255)
                    confirmed_count += 1
                elif final_state == "POSSIBLE_PILOT":
                    color = (0, 165, 255)
                    possible_count += 1
                else:
                    color = (0, 200, 0)

                if (
                    record.controller_evaluated
                    and global_controller_box is not None
                    and evidence.level != "negative"
                    and detection.score >= args.controller_draw_thr
                ):
                    draw_controller_box(
                        display_frame, global_controller_box, detection.score
                    )

                hits = controller_result.positive if controller_result else 0
                samples = controller_result.total if controller_result else 0
                label_lines = [
                    (
                        f"P{record.track_id:03d} {final_state} "
                        f"score={fusion_result.smooth_score:.2f}"
                    ),
                    (
                        f"C={fusion_result.controller_strength:.2f} "
                        f"H={fusion_result.controller_hit_ratio:.2f} "
                        f"P={fusion_result.pose_score:.2f} "
                        f"A={fusion_result.association_score:.2f}"
                    ),
                    (
                        f"ctrl={detection.score:.2f}/{evidence.level} "
                        f"hit={fusion_result.controller_hits}/"
                        f"{fusion_result.controller_samples}"
                    ),
                ]
                # Keep ordinary pedestrians visually clean. A confirmed pilot
                # always gets the Body26 skeleton; --draw-keypoints remains the
                # all-person debug override. Draw the skeleton first so the
                # opaque label background stays readable on top of it.
                draw_person_keypoints = bool(
                    args.draw_keypoints
                    or final_state == "CONFIRMED_PILOT"
                )
                if draw_person_keypoints:
                    pose_app.draw_pose(
                        display_frame,
                        record.keypoints,
                        record.keypoint_scores,
                        args.keypoint_thr,
                    )
                pose_app.draw_label(display_frame, record.bbox, label_lines, color)

                people_payload.append(
                    {
                        "track_id": record.track_id,
                        "bbox_xyxy": rounded_box(record.bbox),
                        "keypoints": pose_app.serialize_keypoints(
                            record.keypoints, record.keypoint_scores
                        ),
                        "keypoints_drawn": draw_person_keypoints,
                        "detection_score": round(record.detection_score, 4),
                        "pose_state": record.pose_temporal.state,
                        "pose_candidate_confidence": round(
                            float(record.pose_temporal.confidence), 4
                        ),
                        "pose_candidate_level": record.pose_features.candidate_level,
                        "pose_frame_candidate": bool(
                            record.pose_features.frame_candidate
                        ),
                        "pose_holding_score": round(
                            float(record.pose_features.holding_score), 4
                        ),
                        "controller_roi_candidate": record.offer_controller_roi,
                        "controller_evaluated": record.controller_evaluated,
                        "controller_score": round(float(detection.score), 4)
                        if record.controller_evaluated
                        else None,
                        "controller_evidence_score": round(float(evidence.score), 4)
                        if record.controller_evaluated
                        else None,
                        "controller_evidence_level": evidence.level
                        if record.controller_evaluated
                        else None,
                        "controller_evidence_reason": evidence.reason
                        if record.controller_evaluated
                        else None,
                        "controller_spatial_match": association.accepted
                        if record.controller_evaluated
                        else None,
                        "controller_association_reason": association.reason
                        if record.controller_evaluated
                        else None,
                        "controller_wrist_distance_ratio": round(
                            float(association.wrist_distance_ratio), 4
                        ) if association.wrist_distance_ratio is not None else None,
                        "controller_broad_pose_candidate": broad_pose_candidate,
                        "controller_bbox_roi": rounded_box(detection.bbox_roi)
                        if record.controller_evaluated
                        else None,
                        "controller_bbox_global": rounded_box(
                            global_controller_box
                        ),
                        "controller_vote": [hits, samples],
                        "controller_temporal_confirmed": confirmed_now,
                        "controller_confirmed_with_hold": controller_confirmed,
                        "pilot_score_raw": round(fusion_result.raw_score, 4),
                        "pilot_score_smooth": round(
                            fusion_result.smooth_score, 4
                        ),
                        "pilot_controller_strength": round(
                            fusion_result.controller_strength, 4
                        ),
                        "pilot_controller_hit_ratio": round(
                            fusion_result.controller_hit_ratio, 4
                        ),
                        "pilot_pose_score": round(fusion_result.pose_score, 4),
                        "pilot_pose_ratio": round(fusion_result.pose_ratio, 4),
                        "pilot_pose_mean": round(fusion_result.pose_mean, 4),
                        "pilot_association_score": round(
                            fusion_result.association_score, 4
                        ),
                        "pilot_controller_hits": fusion_result.controller_hits,
                        "pilot_controller_samples": (
                            fusion_result.controller_samples
                        ),
                        "pilot_controller_strong_hits": fusion_result.strong_hits,
                        "pilot_confirm_gate": fusion_result.confirm_gate,
                        "pilot_decision_reason": fusion_result.reason,
                        "final_pilot_state": final_state,
                    }
                )

            pose_temporal.prune(
                frame_index,
                timestamp=timeline_seconds,
                max_age_seconds=max(10.0, args.candidate_hold_seconds * 2.0),
            )
            controller_temporal.prune(
                frame_index,
                max_age_frames=max(90, args.track_max_age * 3),
                timestamp=timeline_seconds,
                max_age_seconds=max(10.0, args.pilot_hold_seconds * 4.0),
            )
            pilot_fusion.prune(
                frame_index,
                max_age_frames=max(90, args.track_max_age * 3),
                timestamp=timeline_seconds,
                max_age_seconds=max(
                    10.0,
                    args.fusion_controller_stale_seconds * 4.0,
                ),
            )
            stale_ids = [
                track_id
                for track_id, last_seen in last_seen_at.items()
                if timeline_seconds - last_seen > 10.0
            ]
            for track_id in stale_ids:
                last_seen_at.pop(track_id, None)
                last_confirmed_at.pop(track_id, None)

            elapsed = max(1e-6, time.perf_counter() - started)
            instant_fps = 1.0 / elapsed
            fps_ema = (
                instant_fps
                if fps_ema == 0.0
                else 0.9 * fps_ema + 0.1 * instant_fps
            )
            cv2.putText(
                display_frame,
                (
                    f"POSE+CTRL FPS {fps_ema:.1f} | frame {frame_index} | "
                    f"pilot {confirmed_count} | possible {possible_count}"
                    + (
                        f" | dropped {dropped_frames}"
                        if live_source
                        else ""
                    )
                ),
                (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if json_file is not None:
                json_file.write(
                    json.dumps(
                        {
                            "schema_version": "3.0-confidence-fusion",
                            "source_video": str(args.input),
                            "live_source": live_source,
                            "dropped_source_frames": dropped_frames,
                            "frame_index": frame_index,
                            "processed_frame": processed_frames,
                            "timestamp_ms": round(timeline_seconds * 1000.0, 1),
                            "fps": round(effective_fps, 6),
                            "image_width": width,
                            "image_height": height,
                            "raw_detection_count": len(raw_people),
                            "quality_rejections": rejection_counts,
                            "people": people_payload,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                if processed_frames % 20 == 0:
                    json_file.flush()

            if writer is not None:
                writer.write(display_frame)
            if not args.no_show:
                cv2.imshow("UAV pilot: pose + controller", display_frame)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
            progress.update(frame_index)
    finally:
        body_failed = sys.exc_info()[0] is not None
        cleanup_error: Optional[BaseException] = None
        progress.close(frame_index)
        if live_reader is not None:
            live_reader.close()
        else:
            capture.release()
        if json_file is not None:
            json_file.close()
        if writer is not None:
            try:
                writer.release()
            except BaseException as error:
                cleanup_error = error
        cv2.destroyAllWindows()
        if cleanup_error is not None and not body_failed:
            raise cleanup_error

    print(
        f"Done. Processed {processed_frames} pose/controller frames; "
        f"dropped {dropped_frames} stale live frames."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


