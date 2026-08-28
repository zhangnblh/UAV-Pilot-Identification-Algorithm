from __future__ import annotations

import unittest

import numpy as np

from pilot_controller_video_confidence import (
    ControllerTemporalConfirmer,
    PilotConfidenceFusion,
    associate_controller_to_person,
    classify_controller_evidence,
    extract_best_controller,
    square_person_roi,
)


class RoiTests(unittest.TestCase):
    def test_square_roi_and_global_mapping(self) -> None:
        frame = np.zeros((200, 300, 3), dtype=np.uint8)
        roi, transform = square_person_roi(
            frame, [100, 20, 200, 180], output_size=640
        )
        self.assertEqual(roi.shape, (640, 640, 3))
        mapped = transform.roi_box_to_frame([0, 0, 640, 640], 300, 200)
        expected = [
            max(0.0, transform.crop_xyxy[0]),
            max(0.0, transform.crop_xyxy[1]),
            min(300.0, transform.crop_xyxy[2]),
            min(200.0, transform.crop_xyxy[3]),
        ]
        np.testing.assert_allclose(mapped, expected, atol=1e-6)

    def test_out_of_frame_roi_uses_padding(self) -> None:
        frame = np.full((100, 100, 3), 25, dtype=np.uint8)
        roi, _ = square_person_roi(frame, [-20, -10, 40, 90], output_size=640)
        self.assertEqual(roi.shape, (640, 640, 3))
        self.assertTrue(np.any(roi == 114))


class TemporalTests(unittest.TestCase):
    def test_three_hits_in_full_five_sample_window(self) -> None:
        confirmer = ControllerTemporalConfirmer(
            window_size=5,
            min_positive=3,
            score_threshold=0.30,
            require_full_window=True,
        )
        results = [
            confirmer.update(3, score, frame_index=index)
            for index, score in enumerate([0.4, 0.1, 0.5, 0.1, 0.6], 1)
        ]
        self.assertFalse(results[3].confirmed)
        self.assertTrue(results[4].confirmed)
        self.assertEqual((results[4].positive, results[4].total), (3, 5))


class ControllerEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.person_box = np.asarray([100, 50, 300, 450], dtype=np.float32)
        self.keypoints = np.zeros((26, 2), dtype=np.float32)
        self.scores = np.zeros(26, dtype=np.float32)
        self.keypoints[9] = [180, 240]
        self.keypoints[10] = [220, 240]
        self.scores[[9, 10]] = 0.9

    def test_controller_between_wrists_is_associated(self) -> None:
        result = associate_controller_to_person(
            [175, 220, 225, 260], self.person_box, self.keypoints, self.scores
        )
        self.assertTrue(result.accepted)
        self.assertEqual(result.reason, "near_wrists")

    def test_controller_outside_person_zone_is_rejected(self) -> None:
        result = associate_controller_to_person(
            [330, 210, 390, 270], self.person_box, self.keypoints, self.scores
        )
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "outside_person_zone")

    def test_weak_controller_requires_pose(self) -> None:
        accepted = classify_controller_evidence(0.40, True, True)
        rejected = classify_controller_evidence(0.40, True, False)
        self.assertEqual(accepted.level, "broad")
        self.assertEqual(rejected.level, "negative")

    def test_parser_selects_controller_class(self) -> None:
        result = {
            "pred_instances": {
                "bboxes": np.asarray(
                    [[1, 2, 10, 20], [3, 4, 30, 40], [5, 6, 50, 60]],
                    dtype=np.float32,
                ),
                "scores": np.asarray([0.25, 0.91, 0.70], dtype=np.float32),
                "labels": np.asarray([0, 1, 0], dtype=np.int64),
            }
        }
        detection = extract_best_controller(result)
        self.assertAlmostEqual(detection.score, 0.70, places=5)
        np.testing.assert_allclose(detection.bbox_roi, [5, 6, 50, 60])


class FusionValidationTests(unittest.TestCase):
    def test_zero_weight_sum_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PilotConfidenceFusion(
                controller_strength_weight=0.0,
                controller_temporal_weight=0.0,
                pose_weight=0.0,
                association_weight=0.0,
            )


if __name__ == "__main__":
    unittest.main()
