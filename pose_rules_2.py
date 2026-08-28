"""Scale-aware posture evidence for preliminary UAV-pilot screening.

This module keeps the public interface expected by ``app.py`` while separating
strong posture candidates from weak/low-resolution evidence.  Posture is only
one source of evidence; no score in this module is a pilot identity probability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

import pose_rules_1 as _base


# Re-export the common Body26 indices and video utilities so app.py can switch
# between pose_rules_1 and pose_rules_2 by changing only its import module.
NOSE = _base.NOSE
LEFT_EYE = _base.LEFT_EYE
RIGHT_EYE = _base.RIGHT_EYE
LEFT_EAR = _base.LEFT_EAR
RIGHT_EAR = _base.RIGHT_EAR
LEFT_SHOULDER = _base.LEFT_SHOULDER
RIGHT_SHOULDER = _base.RIGHT_SHOULDER
LEFT_ELBOW = _base.LEFT_ELBOW
RIGHT_ELBOW = _base.RIGHT_ELBOW
LEFT_WRIST = _base.LEFT_WRIST
RIGHT_WRIST = _base.RIGHT_WRIST
LEFT_HIP = _base.LEFT_HIP
RIGHT_HIP = _base.RIGHT_HIP
HEAD = _base.HEAD
NECK = _base.NECK

DetectionQualityConfig = _base.DetectionQualityConfig
DetectionQuality = _base.DetectionQuality
evaluate_detection_quality = _base.evaluate_detection_quality
bbox_iou = _base.bbox_iou
ByteTrackLite = _base.ByteTrackLite
SimpleIoUTracker = _base.SimpleIoUTracker
TemporalResult = _base.TemporalResult
PilotStateMachine = _base.PilotStateMachine
TemporalVote = _base.TemporalVote


@dataclass(frozen=True)
class RuleConfig:
    keypoint_threshold: float = 0.35
    min_core_mean_confidence: float = 0.45
    strong_core_mean_confidence: float = 0.55

    # Observability and distance-aware reliability. Person-height ratios make
    # the policy portable across 720p/1080p/4K inputs. Shoulder pixels remain a
    # safeguard against unstable wrist geometry and extreme side foreshortening.
    min_shoulder_width: float = 12.0
    strong_shoulder_width: float = 18.0
    min_person_height_ratio: float = 0.055
    strong_person_height_ratio: float = 0.090

    # Strong pattern A: frontal/low controller operation.
    wrist_distance_min: float = 0.45
    wrist_distance_max: float = 1.10
    wrist_height_diff_max: float = 0.32
    wrist_center_x_max: float = 0.85
    wrist_torso_ratio_min: float = 0.35
    wrist_torso_ratio_max: float = 0.95

    # Strong pattern B: oblique/back, high and asymmetric operation. Horizontal
    # centering is intentionally soft because a side-view controller can appear
    # well outside the shoulder midpoint.
    high_wrist_distance_min: float = 0.45
    high_wrist_distance_max: float = 1.20
    # Zero minimum also covers a controller raised symmetrically near the chest;
    # asymmetry is common in side views but is not essential pilot evidence.
    high_wrist_height_diff_min: float = 0.00
    high_wrist_height_diff_max: float = 0.85
    high_wrist_torso_ratio_min: float = -0.20
    high_wrist_torso_ratio_max: float = 0.35
    high_wrist_center_soft_max: float = 2.00

    # Loose pattern is retained only as weak evidence. It can never create a
    # strong frame candidate by itself.
    loose_wrist_distance_min: float = 0.25
    loose_wrist_distance_max: float = 1.30
    loose_wrist_height_diff_max: float = 0.80
    loose_wrist_center_x_max: float = 1.60
    loose_zone_above_shoulder: float = 0.25
    loose_zone_below_hip: float = 0.20
    loose_min_conditions: int = 4

    elbow_angle_min: float = 45.0
    elbow_angle_max: float = 155.0
    holding_threshold: float = 0.80
    weak_score_cap: float = 0.75
    invalid_geometry_score_cap: float = 0.39
    hands_behind_torso_ratio: float = 0.70
    bbox_edge_margin_ratio: float = 0.005

    # Auxiliary head-pitch diagnostics; never a single-frame gate.
    head_pitch_threshold: float = 0.42
    head_neck_distance_max: float = 0.90
    min_face_scale_ratio: float = 0.08
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

    # Additional scale-aware diagnostics. Existing callers may ignore them.
    shoulder_width_pixels: float = 0.0
    person_height_ratio: Optional[float] = None
    scale_quality: float = 0.0
    scale_level: str = "unobservable"
    loose_condition_count: int = 0
    weak_candidate: bool = False
    strong_candidate: bool = False
    candidate_level: str = "none"

    @property
    def frame_candidate(self) -> bool:
        return (
            self.valid
            and self.strong_candidate
            and self.grip_geometry_valid
            and not self.bbox_truncated
            and not self.hands_behind_suspected
            and self.holding_score >= self.candidate_threshold
        )


def _band_score(value: float, low: float, high: float) -> float:
    """Continuous score with a maximum at the middle of an accepted band."""
    if not low <= value <= high:
        return 0.0
    middle = 0.5 * (low + high)
    half_width = max(1e-6, 0.5 * (high - low))
    return float(np.clip(1.0 - 0.35 * abs(value - middle) / half_width, 0.0, 1.0))


def _upper_body_truncated(
    points: np.ndarray,
    image_shape: Optional[Sequence[int]],
    margin_ratio: float,
) -> bool:
    if image_shape is None or len(image_shape) < 2:
        return False
    height, width = int(image_shape[0]), int(image_shape[1])
    if height <= 0 or width <= 0:
        return False
    margin_x, margin_y = margin_ratio * width, margin_ratio * height
    return bool(
        np.any(points[:, 0] <= margin_x)
        or np.any(points[:, 1] <= margin_y)
        or np.any(points[:, 0] >= width - margin_x)
        or np.any(points[:, 1] >= height - margin_y)
    )


def analyze_pose(
    keypoints: np.ndarray,
    scores: np.ndarray,
    config: RuleConfig,
    bbox: Optional[np.ndarray] = None,
    image_shape: Optional[Sequence[int]] = None,
) -> PoseFeatures:
    """Return strong/weak explainable posture evidence for one detected person."""
    keypoints = np.asarray(keypoints, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    required = (
        LEFT_SHOULDER,
        RIGHT_SHOULDER,
        LEFT_ELBOW,
        RIGHT_ELBOW,
        LEFT_WRIST,
        RIGHT_WRIST,
    )
    if keypoints.ndim != 2 or keypoints.shape[1] < 2:
        return PoseFeatures(False, reason="bad keypoint shape")
    if max(required) >= len(keypoints) or len(scores) < len(keypoints):
        return PoseFeatures(False, reason="model has too few keypoints")
    if not all(_base._valid(scores, index, config.keypoint_threshold) for index in required):
        return PoseFeatures(False, reason="core upper-body keypoints are uncertain")

    core_confidence = float(np.mean(scores[list(required)]))
    if core_confidence < config.min_core_mean_confidence:
        return PoseFeatures(
            False,
            pose_quality=core_confidence,
            reason="core keypoint mean confidence is too low",
        )

    ls = _base._point(keypoints, LEFT_SHOULDER)
    rs = _base._point(keypoints, RIGHT_SHOULDER)
    le = _base._point(keypoints, LEFT_ELBOW)
    re = _base._point(keypoints, RIGHT_ELBOW)
    lw = _base._point(keypoints, LEFT_WRIST)
    rw = _base._point(keypoints, RIGHT_WRIST)
    shoulder_width = float(np.linalg.norm(ls - rs))

    person_height_ratio: Optional[float] = None
    box_height: Optional[float] = None
    if bbox is not None:
        box = np.asarray(bbox, dtype=np.float32).reshape(-1)[:4]
        if len(box) == 4 and np.all(np.isfinite(box)) and box[3] > box[1]:
            box_height = float(box[3] - box[1])
            if image_shape is not None and len(image_shape) >= 2 and image_shape[0] > 0:
                person_height_ratio = box_height / float(image_shape[0])

    if shoulder_width < config.min_shoulder_width:
        return PoseFeatures(
            False,
            pose_quality=core_confidence,
            shoulder_width_pixels=shoulder_width,
            person_height_ratio=person_height_ratio,
            reason="person is too small: shoulder width",
        )
    if (
        person_height_ratio is not None
        and person_height_ratio < config.min_person_height_ratio
    ):
        return PoseFeatures(
            False,
            pose_quality=core_confidence,
            shoulder_width_pixels=shoulder_width,
            person_height_ratio=person_height_ratio,
            reason="person is too small: bbox height ratio",
        )

    if person_height_ratio is not None:
        height_scale = float(
            np.clip(
                (person_height_ratio - config.min_person_height_ratio)
                / (config.strong_person_height_ratio - config.min_person_height_ratio),
                0.0,
                1.0,
            )
        )
    else:
        height_scale = 0.0
    shoulder_scale = float(
        np.clip(
            (shoulder_width - config.min_shoulder_width)
            / (config.strong_shoulder_width - config.min_shoulder_width),
            0.0,
            1.0,
        )
    )
    # Bbox height is more stable than apparent shoulder width in a side view.
    scale_quality = height_scale if person_height_ratio is not None else shoulder_scale
    scale_strong = bool(
        person_height_ratio >= config.strong_person_height_ratio
        if person_height_ratio is not None
        else shoulder_width >= config.strong_shoulder_width
    )
    scale_level = "reliable" if scale_strong else "weak"

    wrist_center = (lw + rw) / 2.0
    shoulder_center = (ls + rs) / 2.0
    hips_available = bool(
        LEFT_HIP < len(keypoints)
        and RIGHT_HIP < len(keypoints)
        and _base._valid(scores, LEFT_HIP, config.keypoint_threshold)
        and _base._valid(scores, RIGHT_HIP, config.keypoint_threshold)
    )
    if hips_available:
        hip_center = (
            _base._point(keypoints, LEFT_HIP)
            + _base._point(keypoints, RIGHT_HIP)
        ) / 2.0
        torso_height = max(
            0.5 * shoulder_width,
            float(abs(hip_center[1] - shoulder_center[1])),
        )
        lower_reference_y = float(hip_center[1])
    else:
        upper_arm_length = 0.5 * (
            float(np.linalg.norm(ls - le)) + float(np.linalg.norm(rs - re))
        )
        estimated_torso = max(1.7 * shoulder_width, 2.2 * upper_arm_length)
        if box_height is not None:
            estimated_torso = max(estimated_torso, 0.38 * box_height)
        torso_height = estimated_torso
        lower_reference_y = float(shoulder_center[1] + torso_height)

    wrist_distance_ratio = float(np.linalg.norm(lw - rw) / shoulder_width)
    wrist_height_diff_ratio = float(abs(lw[1] - rw[1]) / shoulder_width)
    wrist_torso_ratio = float(
        (wrist_center[1] - shoulder_center[1]) / max(1.0, torso_height)
    )
    center_offset_ratio = abs(float(wrist_center[0] - shoulder_center[0])) / shoulder_width

    hands_close = bool(
        config.wrist_distance_min < wrist_distance_ratio < config.wrist_distance_max
    )
    hands_level = bool(wrist_height_diff_ratio < config.wrist_height_diff_max)
    in_vertical_zone = bool(
        config.wrist_torso_ratio_min
        < wrist_torso_ratio
        < config.wrist_torso_ratio_max
    )
    in_center_zone = bool(center_offset_ratio < config.wrist_center_x_max)

    left_elbow_angle = _base._angle_deg(ls, le, lw)
    right_elbow_angle = _base._angle_deg(rs, re, rw)
    elbows_bent = bool(
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

    front_operation_pose = bool(hands_close and hands_level and in_vertical_zone)
    high_distance_ok = bool(
        config.high_wrist_distance_min
        < wrist_distance_ratio
        < config.high_wrist_distance_max
    )
    high_height_ok = bool(
        config.high_wrist_height_diff_min
        <= wrist_height_diff_ratio
        < config.high_wrist_height_diff_max
    )
    high_vertical_ok = bool(
        config.high_wrist_torso_ratio_min
        < wrist_torso_ratio
        < config.high_wrist_torso_ratio_max
    )
    high_operation_pose = bool(
        high_distance_ok and high_height_ok and high_vertical_ok and elbows_bent
    )

    front_score = (
        0.35 * _band_score(
            wrist_distance_ratio,
            config.wrist_distance_min,
            config.wrist_distance_max,
        )
        + 0.25 * float(hands_level)
        + 0.20 * float(in_vertical_zone)
        + 0.10 * float(in_center_zone)
        + 0.10 * float(elbows_bent or forearms_converging)
    )
    high_center_score = float(
        np.clip(1.0 - center_offset_ratio / config.high_wrist_center_soft_max, 0.0, 1.0)
    )
    high_score = (
        0.30 * _band_score(
            wrist_distance_ratio,
            config.high_wrist_distance_min,
            config.high_wrist_distance_max,
        )
        + 0.20 * float(high_height_ok)
        + 0.20 * float(high_vertical_ok)
        + 0.15 * float(elbows_bent)
        + 0.10 * high_center_score
        + 0.05 * float(forearms_converging)
    )

    loose_conditions = (
        config.loose_wrist_distance_min
        < wrist_distance_ratio
        < config.loose_wrist_distance_max,
        wrist_height_diff_ratio < config.loose_wrist_height_diff_max,
        shoulder_center[1] - config.loose_zone_above_shoulder * shoulder_width
        < wrist_center[1]
        < lower_reference_y + config.loose_zone_below_hip * shoulder_width,
        center_offset_ratio < config.loose_wrist_center_x_max,
        elbows_bent,
    )
    loose_condition_count = int(sum(bool(value) for value in loose_conditions))
    loose_candidate = loose_condition_count >= config.loose_min_conditions

    grip_geometry_valid = bool(front_operation_pose or high_operation_pose)
    raw_score = max(front_score, high_score)
    if loose_candidate and not grip_geometry_valid:
        raw_score = max(raw_score, 0.65)

    upper_points = np.asarray([ls, rs, le, re, lw, rw])
    bbox_truncated = _upper_body_truncated(
        upper_points, image_shape, config.bbox_edge_margin_ratio
    )
    shoulder_horizontal_ratio = float(abs(ls[0] - rs[0]) / shoulder_width)
    back_facing = bool(shoulder_horizontal_ratio >= 0.35 and ls[0] < rs[0])
    hands_behind_suspected = bool(
        back_facing
        and wrist_torso_ratio >= config.hands_behind_torso_ratio
        and in_center_zone
    )

    strong_candidate = bool(
        grip_geometry_valid
        and raw_score >= config.holding_threshold
        and scale_strong
        and core_confidence >= config.strong_core_mean_confidence
        and not bbox_truncated
        and not hands_behind_suspected
    )
    weak_candidate = bool(
        not strong_candidate
        and (grip_geometry_valid or loose_candidate)
        and not hands_behind_suspected
    )
    penalty_reasons = []
    if not grip_geometry_valid:
        penalty_reasons.append("loose_geometry_only")
    if not scale_strong:
        penalty_reasons.append("weak_person_scale")
    if core_confidence < config.strong_core_mean_confidence:
        penalty_reasons.append("weak_pose_quality")
    if bbox_truncated:
        penalty_reasons.append("upper_body_truncated")
    if hands_behind_suspected:
        penalty_reasons.append("hands_behind_suspected")

    holding_score = float(np.clip(raw_score, 0.0, 1.0))
    if weak_candidate:
        holding_score = min(holding_score, config.weak_score_cap)
    if not strong_candidate and not weak_candidate:
        holding_score = min(holding_score, config.invalid_geometry_score_cap)
    if bbox_truncated or hands_behind_suspected:
        holding_score = min(holding_score, config.invalid_geometry_score_cap)

    if strong_candidate:
        candidate_level = "strong"
        operation_pose_type = (
            "front_low" if front_score >= high_score else "oblique_high"
        )
    elif weak_candidate:
        candidate_level = "weak"
        operation_pose_type = (
            "front_low_weak"
            if front_operation_pose and front_score >= high_score
            else "oblique_high_weak"
            if high_operation_pose
            else "wide_loose"
        )
    else:
        candidate_level = "none"
        operation_pose_type = "none"

    head_down, head_pitch_score, head_confidence, head_gap_ratio = (
        _base._estimate_head_pitch(keypoints, scores, shoulder_width, config)
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
        hands_close=hands_close,
        hands_level=hands_level,
        in_vertical_zone=in_vertical_zone,
        in_center_zone=in_center_zone,
        elbows_bent=elbows_bent,
        forearms_converging=forearms_converging,
        front_operation_pose=front_operation_pose,
        high_operation_pose=high_operation_pose,
        operation_pose_type=operation_pose_type,
        bbox_truncated=bbox_truncated,
        back_facing=back_facing,
        hands_behind_suspected=hands_behind_suspected,
        penalty_reason=",".join(penalty_reasons),
        shoulder_width_pixels=shoulder_width,
        person_height_ratio=person_height_ratio,
        scale_quality=scale_quality,
        scale_level=scale_level,
        loose_condition_count=loose_condition_count,
        weak_candidate=weak_candidate,
        strong_candidate=strong_candidate,
        candidate_level=candidate_level,
    )
