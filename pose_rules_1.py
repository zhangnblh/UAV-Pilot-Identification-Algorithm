from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# Halpe/RTMPose Body26 keypoint indices.
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
HEAD = 17
NECK = 18


@dataclass(frozen=True)
class RuleConfig:
    keypoint_threshold: float = 0.35
    min_core_mean_confidence: float = 0.45
    min_shoulder_width: float = 12.0
    wrist_distance_min: float = 0.15
    wrist_distance_max: float = 1.30
    wrist_height_diff_max: float = 0.70
    wrist_center_x_max: float = 0.70
    wrist_zone_above_shoulder: float = 0.20
    wrist_zone_below_hip: float = 0.30
    elbow_angle_min: float = 45.0
    elbow_angle_max: float = 155.0
    holding_threshold: float = 0.80
    # 2D multi-landmark head-pitch proxy. It is auxiliary evidence, not a gate.
    head_pitch_threshold: float = 0.42
    head_neck_distance_max: float = 0.90
    min_face_scale_ratio: float = 0.08
    # Kept for command/config compatibility; no longer used for orientation.
    head_gap_threshold: float = 1.00
    head_gap_mode: str = "high"


@dataclass
class PoseFeatures:
    valid: bool
    holding_score: float = 0.0
    candidate_threshold: float = 0.80
    pose_quality: float = 0.0
    hips_available: bool = False
    head_down: Optional[bool] = None
    head_pitch_score: Optional[float] = None
    head_state_confidence: float = 0.0
    head_gap_ratio: Optional[float] = None
    wrist_distance_ratio: Optional[float] = None
    wrist_height_diff_ratio: Optional[float] = None
    wrist_torso_ratio: Optional[float] = None
    left_elbow_angle: Optional[float] = None
    right_elbow_angle: Optional[float] = None
    grip_geometry_valid: bool = False
    hands_close: bool = False
    hands_level: bool = False
    in_vertical_zone: bool = False
    in_center_zone: bool = False
    elbows_bent: bool = False
    forearms_converging: bool = False
    front_operation_pose: bool = False
    high_operation_pose: bool = False
    operation_pose_type: str = "none"
    bbox_truncated: bool = False
    back_facing: bool = False
    hands_behind_suspected: bool = False
    penalty_reason: str = ""
    reason: str = ""

    @property
    def frame_candidate(self) -> bool:
        # Head direction is intentionally not a necessary condition.
        return self.valid and self.holding_score >= self.candidate_threshold


def _point(keypoints: np.ndarray, index: int) -> np.ndarray:
    return np.asarray(keypoints[index, :2], dtype=np.float32)


def _valid(scores: np.ndarray, index: int, threshold: float) -> bool:
    return (
        index < len(scores)
        and np.isfinite(scores[index])
        and float(scores[index]) >= threshold
    )


def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Optional[float]:
    ba = a - b
    bc = c - b
    denominator = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if denominator < 1e-6:
        return None
    cosine = float(np.dot(ba, bc) / denominator)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _estimate_head_pitch(
    keypoints: np.ndarray,
    scores: np.ndarray,
    shoulder_width: float,
    config: RuleConfig,
) -> Tuple[Optional[bool], Optional[float], float, Optional[float]]:
    """Estimate a 2D head-pitch proxy from face landmarks.

    Both eyes and the nose are required. The nose displacement is measured
    perpendicular to the eye line, which makes the signal less sensitive to
    in-plane head roll. Neck distance is used only as a plausibility check.
    """
    face_required = (NOSE, LEFT_EYE, RIGHT_EYE)
    if not all(
        index < len(keypoints)
        and _valid(scores, index, config.keypoint_threshold)
        for index in face_required
    ):
        return None, None, 0.0, None

    nose = _point(keypoints, NOSE)
    left_eye = _point(keypoints, LEFT_EYE)
    right_eye = _point(keypoints, RIGHT_EYE)
    eye_vector = right_eye - left_eye
    eye_distance = float(np.linalg.norm(eye_vector))
    if eye_distance < config.min_face_scale_ratio * shoulder_width:
        return None, None, 0.0, None

    face_scale = eye_distance
    if (
        _valid(scores, LEFT_EAR, config.keypoint_threshold)
        and _valid(scores, RIGHT_EAR, config.keypoint_threshold)
    ):
        ear_distance = float(
            np.linalg.norm(
                _point(keypoints, RIGHT_EAR) - _point(keypoints, LEFT_EAR)
            )
        )
        if ear_distance >= eye_distance:
            face_scale = 0.5 * (eye_distance + ear_distance)

    eye_center = (left_eye + right_eye) / 2.0
    downward_normal = np.asarray([-eye_vector[1], eye_vector[0]], dtype=np.float32)
    if downward_normal[1] < 0:
        downward_normal *= -1.0
    downward_normal /= max(1e-6, float(np.linalg.norm(downward_normal)))
    nose_drop = float(np.dot(nose - eye_center, downward_normal) / face_scale)

    head_gap_ratio: Optional[float] = None
    neck_plausible = True
    neck_confidence = 0.0
    if NECK < len(keypoints) and _valid(scores, NECK, config.keypoint_threshold):
        head_gap_ratio = float(
            np.linalg.norm(nose - _point(keypoints, NECK)) / shoulder_width
        )
        neck_plausible = head_gap_ratio <= config.head_neck_distance_max
        neck_confidence = float(np.clip(scores[NECK], 0.0, 1.0))

    face_confidence = float(
        np.mean([scores[NOSE], scores[LEFT_EYE], scores[RIGHT_EYE]])
    )
    state_confidence = float(
        np.clip(0.8 * face_confidence + 0.2 * neck_confidence, 0.0, 1.0)
    )
    return (
        bool(nose_drop >= config.head_pitch_threshold and neck_plausible),
        nose_drop,
        state_confidence,
        head_gap_ratio,
    )


def analyze_pose(
    keypoints: np.ndarray,
    scores: np.ndarray,
    config: RuleConfig,
    bbox: Optional[np.ndarray] = None,
    image_shape: Optional[Sequence[int]] = None,
) -> PoseFeatures:
    """Extract explainable two-hand-operation and auxiliary head-pitch evidence."""
    keypoints = np.asarray(keypoints, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)

    core_required = (
        LEFT_SHOULDER,
        RIGHT_SHOULDER,
        LEFT_ELBOW,
        RIGHT_ELBOW,
        LEFT_WRIST,
        RIGHT_WRIST,
    )
    if keypoints.ndim != 2 or keypoints.shape[1] < 2:
        return PoseFeatures(False, reason="bad keypoint shape")
    if max(core_required) >= len(keypoints) or len(scores) < len(keypoints):
        return PoseFeatures(False, reason="model has too few keypoints")
    if not all(
        _valid(scores, index, config.keypoint_threshold)
        for index in core_required
    ):
        return PoseFeatures(False, reason="core upper-body keypoints are uncertain")

    core_confidence = float(np.mean(scores[list(core_required)]))
    if core_confidence < config.min_core_mean_confidence:
        return PoseFeatures(
            False,
            pose_quality=core_confidence,
            reason="core keypoint mean confidence is too low",
        )

    ls, rs = _point(keypoints, LEFT_SHOULDER), _point(keypoints, RIGHT_SHOULDER)
    le, re = _point(keypoints, LEFT_ELBOW), _point(keypoints, RIGHT_ELBOW)
    lw, rw = _point(keypoints, LEFT_WRIST), _point(keypoints, RIGHT_WRIST)
    shoulder_width = float(np.linalg.norm(ls - rs))
    if shoulder_width < config.min_shoulder_width:
        return PoseFeatures(
            False,
            pose_quality=core_confidence,
            reason="person is too small",
        )

    wrist_center = (lw + rw) / 2.0
    shoulder_center = (ls + rs) / 2.0
    hips_available = (
        LEFT_HIP < len(keypoints)
        and RIGHT_HIP < len(keypoints)
        and _valid(scores, LEFT_HIP, config.keypoint_threshold)
        and _valid(scores, RIGHT_HIP, config.keypoint_threshold)
    )
    if hips_available:
        lower_body_reference_y = float(
            (_point(keypoints, LEFT_HIP)[1] + _point(keypoints, RIGHT_HIP)[1])
            / 2.0
        )
    else:
        upper_arm_length = 0.5 * (
            float(np.linalg.norm(ls - le)) + float(np.linalg.norm(rs - re))
        )
        estimated_torso = max(1.7 * shoulder_width, 2.2 * upper_arm_length)
        if bbox is not None:
            box = np.asarray(bbox, dtype=np.float32).reshape(-1)[:4]
            if len(box) == 4 and box[3] > box[1]:
                estimated_torso = max(
                    estimated_torso, 0.38 * float(box[3] - box[1])
                )
        lower_body_reference_y = float(shoulder_center[1] + estimated_torso)

    wrist_distance_ratio = float(np.linalg.norm(lw - rw) / shoulder_width)
    wrist_height_diff_ratio = float(abs(lw[1] - rw[1]) / shoulder_width)
    torso_height = max(
        0.5 * shoulder_width,
        float(abs(lower_body_reference_y - shoulder_center[1])),
    )
    wrist_torso_ratio = float(
        (wrist_center[1] - shoulder_center[1]) / torso_height
    )

    hands_close = (
        config.wrist_distance_min
        < wrist_distance_ratio
        < config.wrist_distance_max
    )
    hands_level = wrist_height_diff_ratio < config.wrist_height_diff_max
    in_vertical_zone = (
        shoulder_center[1] - config.wrist_zone_above_shoulder * shoulder_width
        < wrist_center[1]
        < lower_body_reference_y
        + config.wrist_zone_below_hip * shoulder_width
    )
    in_center_zone = (
        abs(float(wrist_center[0] - shoulder_center[0])) / shoulder_width
        < config.wrist_center_x_max
    )

    left_elbow_angle = _angle_deg(ls, le, lw)
    right_elbow_angle = _angle_deg(rs, re, rw)
    elbows_bent = (
        left_elbow_angle is not None
        and right_elbow_angle is not None
        and config.elbow_angle_min < left_elbow_angle < config.elbow_angle_max
        and config.elbow_angle_min < right_elbow_angle < config.elbow_angle_max
    )

    elbow_distance_ratio = float(np.linalg.norm(le - re) / shoulder_width)
    forearms_converging = bool(
        elbow_distance_ratio > 1e-6
        and wrist_distance_ratio < 0.95 * elbow_distance_ratio
    )

    conditions: Iterable[bool] = (
        hands_close,
        hands_level,
        in_vertical_zone,
        in_center_zone,
        elbows_bent,
    )
    holding_score = float(sum(bool(item) for item in conditions) / 5.0)
    grip_geometry_valid = bool(hands_close and hands_level and in_vertical_zone)

    # Compatibility diagnostics for app.py/detect_image.py.  They deliberately
    # do not alter the original wide five-condition score or frame_candidate.
    bbox_truncated = False
    if image_shape is not None and len(image_shape) >= 2:
        frame_height, frame_width = int(image_shape[0]), int(image_shape[1])
        if frame_height > 0 and frame_width > 0:
            margin_x = 0.005 * frame_width
            margin_y = 0.005 * frame_height
            core_points = np.asarray([ls, rs, le, re, lw, rw])
            bbox_truncated = bool(
                np.any(core_points[:, 0] <= margin_x)
                or np.any(core_points[:, 1] <= margin_y)
                or np.any(core_points[:, 0] >= frame_width - margin_x)
                or np.any(core_points[:, 1] >= frame_height - margin_y)
            )

    shoulder_horizontal_ratio = float(abs(ls[0] - rs[0]) / shoulder_width)
    back_facing = bool(shoulder_horizontal_ratio >= 0.35 and ls[0] < rs[0])
    hands_behind_suspected = bool(
        back_facing and wrist_torso_ratio >= 0.70 and in_center_zone
    )
    operation_pose_type = (
        "wide_loose"
        if holding_score >= config.holding_threshold
        else "none"
    )
    head_down, head_pitch_score, head_confidence, head_gap_ratio = (
        _estimate_head_pitch(keypoints, scores, shoulder_width, config)
    )

    return PoseFeatures(
        valid=True,
        holding_score=holding_score,
        candidate_threshold=config.holding_threshold,
        pose_quality=core_confidence,
        hips_available=hips_available,
        head_down=head_down,
        head_pitch_score=head_pitch_score,
        head_state_confidence=head_confidence,
        head_gap_ratio=head_gap_ratio,
        wrist_distance_ratio=wrist_distance_ratio,
        wrist_height_diff_ratio=wrist_height_diff_ratio,
        wrist_torso_ratio=wrist_torso_ratio,
        left_elbow_angle=left_elbow_angle,
        right_elbow_angle=right_elbow_angle,
        grip_geometry_valid=grip_geometry_valid,
        hands_close=bool(hands_close),
        hands_level=bool(hands_level),
        in_vertical_zone=bool(in_vertical_zone),
        in_center_zone=bool(in_center_zone),
        elbows_bent=bool(elbows_bent),
        forearms_converging=forearms_converging,
        operation_pose_type=operation_pose_type,
        bbox_truncated=bbox_truncated,
        back_facing=back_facing,
        hands_behind_suspected=hands_behind_suspected,
    )


@dataclass(frozen=True)
class DetectionQualityConfig:
    min_detection_score: float = 0.10
    max_box_area_ratio: float = 0.55
    min_box_area_ratio: float = 0.0003
    min_aspect_ratio: float = 0.12
    max_aspect_ratio: float = 1.20
    min_box_height: float = 24.0
    min_valid_keypoints: int = 6
    min_mean_keypoint_score: float = 0.20


@dataclass(frozen=True)
class DetectionQuality:
    valid: bool
    reason: str = ""
    box_area_ratio: float = 0.0
    aspect_ratio: float = 0.0
    valid_keypoints: int = 0
    mean_keypoint_score: float = 0.0


def _inside_polygon(point: Tuple[float, float], polygon: np.ndarray) -> bool:
    x, y = point
    inside = False
    count = len(polygon)
    for index in range(count):
        x1, y1 = polygon[index]
        x2, y2 = polygon[(index + 1) % count]
        crosses = (y1 > y) != (y2 > y)
        if crosses:
            boundary_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < boundary_x:
                inside = not inside
    return inside


def evaluate_detection_quality(
    bbox: np.ndarray,
    detection_score: float,
    keypoint_scores: np.ndarray,
    frame_shape: Sequence[int],
    config: DetectionQualityConfig,
    roi_polygon: Optional[np.ndarray] = None,
) -> DetectionQuality:
    """Reject implausible detections before they can create or update tracks."""
    box = np.asarray(bbox, dtype=np.float32).reshape(-1)[:4]
    scores = np.asarray(keypoint_scores, dtype=np.float32).reshape(-1)
    if len(box) < 4 or not np.all(np.isfinite(box)):
        return DetectionQuality(False, reason="invalid_bbox")
    frame_height, frame_width = int(frame_shape[0]), int(frame_shape[1])
    if frame_height <= 0 or frame_width <= 0:
        return DetectionQuality(False, reason="invalid_frame_shape")

    x1 = float(np.clip(box[0], 0, frame_width))
    y1 = float(np.clip(box[1], 0, frame_height))
    x2 = float(np.clip(box[2], 0, frame_width))
    y2 = float(np.clip(box[3], 0, frame_height))
    width, height = x2 - x1, y2 - y1
    if width <= 1 or height <= 1:
        return DetectionQuality(False, reason="empty_bbox")

    area_ratio = width * height / float(frame_width * frame_height)
    aspect_ratio = width / height
    finite_scores = scores[np.isfinite(scores)]
    valid_keypoints = int(np.sum(finite_scores >= config.min_mean_keypoint_score))
    mean_score = float(np.mean(finite_scores)) if len(finite_scores) else 0.0
    metrics = dict(
        box_area_ratio=area_ratio,
        aspect_ratio=aspect_ratio,
        valid_keypoints=valid_keypoints,
        mean_keypoint_score=mean_score,
    )
    if not np.isfinite(detection_score) or detection_score < config.min_detection_score:
        return DetectionQuality(False, reason="low_detection_score", **metrics)
    if area_ratio > config.max_box_area_ratio:
        return DetectionQuality(False, reason="bbox_too_large", **metrics)
    if area_ratio < config.min_box_area_ratio or height < config.min_box_height:
        return DetectionQuality(False, reason="bbox_too_small", **metrics)
    if not config.min_aspect_ratio <= aspect_ratio <= config.max_aspect_ratio:
        return DetectionQuality(False, reason="implausible_aspect_ratio", **metrics)
    if valid_keypoints < config.min_valid_keypoints:
        return DetectionQuality(False, reason="too_few_valid_keypoints", **metrics)
    if mean_score < config.min_mean_keypoint_score:
        return DetectionQuality(False, reason="low_mean_keypoint_score", **metrics)
    if roi_polygon is not None:
        polygon = np.asarray(roi_polygon, dtype=np.float32).reshape(-1, 2)
        if len(polygon) >= 3 and not _inside_polygon(
            ((x1 + x2) / 2.0, y2), polygon
        ):
            return DetectionQuality(False, reason="outside_roi", **metrics)
    return DetectionQuality(True, **metrics)


def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)[:4]
    b = np.asarray(b, dtype=np.float32).reshape(-1)[:4]
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _center_distance(a: np.ndarray, b: np.ndarray) -> float:
    center_a = (a[:2] + a[2:4]) / 2.0
    center_b = (b[:2] + b[2:4]) / 2.0
    scale = max(
        1.0,
        np.sqrt(max(1.0, float((a[2] - a[0]) * (a[3] - a[1])))),
        np.sqrt(max(1.0, float((b[2] - b[0]) * (b[3] - b[1])))),
    )
    return float(np.linalg.norm(center_a - center_b) / scale)


def _linear_assignment(cost: np.ndarray) -> List[Tuple[int, int]]:
    """Hungarian assignment for a small rectangular cost matrix."""
    cost = np.asarray(cost, dtype=np.float64)
    if cost.size == 0:
        return []
    transposed = cost.shape[0] > cost.shape[1]
    matrix = cost.T if transposed else cost
    rows, columns = matrix.shape
    u = np.zeros(rows + 1, dtype=np.float64)
    v = np.zeros(columns + 1, dtype=np.float64)
    p = np.zeros(columns + 1, dtype=np.int32)
    way = np.zeros(columns + 1, dtype=np.int32)
    for row in range(1, rows + 1):
        p[0] = row
        column0 = 0
        min_values = np.full(columns + 1, np.inf)
        used = np.zeros(columns + 1, dtype=bool)
        while True:
            used[column0] = True
            row0 = p[column0]
            delta = np.inf
            column1 = 0
            for column in range(1, columns + 1):
                if used[column]:
                    continue
                current = matrix[row0 - 1, column - 1] - u[row0] - v[column]
                if current < min_values[column]:
                    min_values[column] = current
                    way[column] = column0
                if min_values[column] < delta:
                    delta = min_values[column]
                    column1 = column
            for column in range(columns + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    min_values[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = way[column0]
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break
    pairs = [(int(p[column] - 1), column - 1) for column in range(1, columns + 1) if p[column]]
    if transposed:
        return [(column, row) for row, column in pairs]
    return pairs


@dataclass
class _Track:
    bbox: np.ndarray
    last_frame: int
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    score: float = 0.0
    hits: int = 1

    def predict(self, frame_index: int) -> np.ndarray:
        elapsed = max(0, frame_index - self.last_frame)
        return self.bbox + self.velocity * elapsed


class ByteTrackLite:
    """CPU-friendly ByteTrack-style tracker with motion and two-stage matching.

    High-confidence detections are associated first; unmatched tracks then get
    a recovery pass over low-confidence detections. Constant-velocity prediction
    and Hungarian matching reduce ID fragmentation without a ReID model.
    """

    def __init__(
        self,
        iou_threshold: float = 0.15,
        max_age: int = 30,
        high_score_threshold: float = 0.50,
        low_score_threshold: float = 0.10,
        center_distance_threshold: float = 1.50,
    ):
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0 and 1")
        if max_age < 1:
            raise ValueError("max_age must be positive")
        if low_score_threshold > high_score_threshold:
            raise ValueError("low_score_threshold must not exceed high_score_threshold")
        if center_distance_threshold <= 0:
            raise ValueError("center_distance_threshold must be positive")
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.high_score_threshold = high_score_threshold
        self.low_score_threshold = low_score_threshold
        self.center_distance_threshold = center_distance_threshold
        self._next_id = 1
        self._tracks: Dict[int, _Track] = {}

    def _match(
        self,
        track_ids: Sequence[int],
        detection_ids: Sequence[int],
        boxes: Sequence[np.ndarray],
        frame_index: int,
        relaxed: bool = False,
    ) -> List[Tuple[int, int]]:
        if not track_ids or not detection_ids:
            return []
        invalid_cost = 1e5
        cost = np.full((len(track_ids), len(detection_ids)), invalid_cost)
        for row, track_id in enumerate(track_ids):
            predicted = self._tracks[track_id].predict(frame_index)
            for column, detection_id in enumerate(detection_ids):
                iou = bbox_iou(predicted, boxes[detection_id])
                distance = _center_distance(predicted, boxes[detection_id])
                minimum_iou = self.iou_threshold * (0.5 if relaxed else 1.0)
                maximum_distance = self.center_distance_threshold * (
                    1.25 if relaxed else 1.0
                )
                if iou >= minimum_iou or distance <= maximum_distance:
                    cost[row, column] = (1.0 - iou) + 0.20 * distance
        matches = []
        for row, column in _linear_assignment(cost):
            if cost[row, column] < invalid_cost:
                matches.append((track_ids[row], detection_ids[column]))
        return matches

    def _update_track(
        self,
        track_id: int,
        bbox: np.ndarray,
        score: float,
        frame_index: int,
    ) -> None:
        track = self._tracks[track_id]
        elapsed = max(1, frame_index - track.last_frame)
        measured_velocity = (bbox - track.bbox) / elapsed
        track.velocity = 0.65 * track.velocity + 0.35 * measured_velocity
        track.bbox = bbox
        track.last_frame = frame_index
        track.score = score
        track.hits += 1

    def update(
        self,
        boxes: Iterable[np.ndarray],
        frame_index: int,
        scores: Optional[Iterable[float]] = None,
    ) -> Dict[int, int]:
        boxes = [np.asarray(box, dtype=np.float32).reshape(-1)[:4] for box in boxes]
        score_values = (
            [1.0] * len(boxes)
            if scores is None
            else [float(value) for value in scores]
        )
        if len(score_values) != len(boxes):
            raise ValueError("scores and boxes must have the same length")

        stale = [
            track_id
            for track_id, track in self._tracks.items()
            if frame_index - track.last_frame > self.max_age
        ]
        for track_id in stale:
            del self._tracks[track_id]

        active_tracks = list(self._tracks)
        high_detections = [
            index
            for index, score in enumerate(score_values)
            if score >= self.high_score_threshold
        ]
        low_detections = [
            index
            for index, score in enumerate(score_values)
            if self.low_score_threshold <= score < self.high_score_threshold
        ]
        result: Dict[int, int] = {}

        first_matches = self._match(
            active_tracks, high_detections, boxes, frame_index
        )
        matched_tracks = set()
        matched_detections = set()
        for track_id, detection_id in first_matches:
            self._update_track(
                track_id, boxes[detection_id], score_values[detection_id], frame_index
            )
            result[detection_id] = track_id
            matched_tracks.add(track_id)
            matched_detections.add(detection_id)

        remaining_tracks = [
            track_id for track_id in active_tracks if track_id not in matched_tracks
        ]
        second_matches = self._match(
            remaining_tracks, low_detections, boxes, frame_index, relaxed=True
        )
        for track_id, detection_id in second_matches:
            self._update_track(
                track_id, boxes[detection_id], score_values[detection_id], frame_index
            )
            result[detection_id] = track_id
            matched_tracks.add(track_id)
            matched_detections.add(detection_id)

        for detection_id in high_detections:
            if detection_id in matched_detections:
                continue
            track_id = self._next_id
            self._next_id += 1
            self._tracks[track_id] = _Track(
                boxes[detection_id], frame_index, score=score_values[detection_id]
            )
            result[detection_id] = track_id
        return result

    def normalized_speed(self, track_id: int) -> float:
        track = self._tracks.get(track_id)
        if track is None:
            return 0.0
        height = max(1.0, float(track.bbox[3] - track.bbox[1]))
        center_velocity = 0.5 * (track.velocity[:2] + track.velocity[2:4])
        return float(np.linalg.norm(center_velocity) / height)


# Backward-compatible import name. The implementation is no longer IoU-only.
SimpleIoUTracker = ByteTrackLite


@dataclass(frozen=True)
class TemporalResult:
    state: str
    stable: bool
    confidence: float
    positive: int
    total: int
    head_switches: int


@dataclass
class _PilotTrackState:
    history: Deque[bool]
    state: str = "normal"
    consecutive_negative: int = 0
    confidence: float = 0.0
    last_seen: int = 0
    last_seen_time: Optional[float] = None
    negative_since: Optional[float] = None
    stable_head: Optional[bool] = None
    pending_head: Optional[bool] = None
    pending_head_count: int = 0
    head_switches: int = 0


class PilotStateMachine:
    """Track-level evidence fusion with entry/exit hysteresis."""

    def __init__(
        self,
        history_size: int = 12,
        min_positive: int = 8,
        exit_frames: int = 125,
        head_debounce_frames: int = 3,
        exit_seconds: Optional[float] = None,
    ):
        if history_size < 1:
            raise ValueError("history_size must be positive")
        if not 1 <= min_positive <= history_size:
            raise ValueError("min_positive must be within history_size")
        if exit_frames < 1:
            raise ValueError("exit_frames must be positive")
        if exit_seconds is not None and exit_seconds <= 0:
            raise ValueError("exit_seconds must be positive")
        self.history_size = history_size
        self.min_positive = min_positive
        self.exit_frames = exit_frames
        self.exit_seconds = exit_seconds
        self.head_debounce_frames = max(1, head_debounce_frames)
        self._tracks: Dict[int, _PilotTrackState] = {}

    def _update_head(
        self, state: _PilotTrackState, head_down: Optional[bool]
    ) -> None:
        if head_down is None:
            return
        if state.stable_head is None:
            state.stable_head = head_down
            return
        if head_down == state.stable_head:
            state.pending_head = None
            state.pending_head_count = 0
            return
        if state.pending_head == head_down:
            state.pending_head_count += 1
        else:
            state.pending_head = head_down
            state.pending_head_count = 1
        if state.pending_head_count >= self.head_debounce_frames:
            state.stable_head = head_down
            state.head_switches += 1
            state.pending_head = None
            state.pending_head_count = 0

    def update(
        self,
        track_id: int,
        evidence: bool,
        frame_index: int,
        evidence_score: float = 0.0,
        head_down: Optional[bool] = None,
        timestamp: Optional[float] = None,
    ) -> TemporalResult:
        state = self._tracks.get(track_id)
        if state is None:
            state = _PilotTrackState(deque(maxlen=self.history_size))
            self._tracks[track_id] = state
        state.history.append(bool(evidence))
        state.last_seen = frame_index
        state.last_seen_time = timestamp
        self._update_head(state, head_down)

        score = float(np.clip(evidence_score, 0.0, 1.0))
        alpha = 0.18 if evidence else 0.025
        state.confidence = (1.0 - alpha) * state.confidence + alpha * score
        positive = int(sum(state.history))

        if evidence:
            state.consecutive_negative = 0
            state.negative_since = None
            if state.state == "normal":
                state.state = "observing"
            if positive >= self.min_positive:
                state.state = "candidate"
                state.confidence = max(
                    state.confidence, self.min_positive / self.history_size
                )
        else:
            state.consecutive_negative += 1
            if timestamp is not None and state.negative_since is None:
                state.negative_since = timestamp
            expired_by_time = (
                self.exit_seconds is not None
                and timestamp is not None
                and state.negative_since is not None
                and timestamp - state.negative_since >= self.exit_seconds
            )
            expired_by_frames = (
                (self.exit_seconds is None or timestamp is None)
                and state.consecutive_negative >= self.exit_frames
            )
            if (
                state.state == "observing"
                and positive == 0
                and len(state.history) == self.history_size
            ):
                state.state = "normal"
            elif (
                state.state == "candidate"
                and (expired_by_time or expired_by_frames)
            ):
                state.state = "normal"
                state.confidence = min(state.confidence, 0.35)

        return TemporalResult(
            state=state.state,
            stable=state.state == "candidate",
            confidence=float(np.clip(state.confidence, 0.0, 1.0)),
            positive=positive,
            total=len(state.history),
            head_switches=state.head_switches,
        )

    def prune(
        self,
        frame_index: int,
        max_age: int = 300,
        timestamp: Optional[float] = None,
        max_age_seconds: Optional[float] = None,
    ) -> None:
        if timestamp is not None and max_age_seconds is not None:
            stale = [
                track_id
                for track_id, state in self._tracks.items()
                if state.last_seen_time is not None
                and timestamp - state.last_seen_time > max_age_seconds
            ]
        else:
            stale = [
                track_id
                for track_id, state in self._tracks.items()
                if frame_index - state.last_seen > max_age
            ]
        for track_id in stale:
            self._tracks.pop(track_id, None)


class TemporalVote:
    """Compatibility wrapper for callers that still expect a three-item vote."""

    def __init__(self, history_size: int = 12, min_positive: int = 8):
        self._machine = PilotStateMachine(
            history_size=history_size,
            min_positive=min_positive,
            exit_frames=history_size,
        )

    def update(
        self, track_id: int, value: bool, frame_index: int
    ) -> Tuple[bool, int, int]:
        result = self._machine.update(
            track_id,
            value,
            frame_index,
            evidence_score=1.0 if value else 0.0,
        )
        return result.stable, result.positive, result.total

    def prune(self, frame_index: int, max_age: int = 90) -> None:
        self._machine.prune(frame_index, max_age)
