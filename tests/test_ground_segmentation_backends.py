import unittest
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ground_segmentation_backends import (
    DenseMajorityGroundSegmenter,
    GroundSegmentationResult,
)


class _FakeDenseBackend:
    def __init__(self, class_maps):
        self.class_maps = [np.asarray(value, np.int16) for value in class_maps]
        self.id2label = {0: "wall", 1: "floor, flooring", 2: "stairs, steps"}

    def predict_batch(self, rgbs):
        self.last_batch_size = len(rgbs)
        outputs = []
        for class_map in self.class_maps:
            mask = np.isin(class_map, [1, 2])
            outputs.append(GroundSegmentationResult(
                mask=mask, class_map=class_map, labels=[], records=[]))
        return outputs


class DenseMajorityGroundSegmenterTest(unittest.TestCase):
    def test_requires_two_of_three_ground_votes(self):
        maps = [
            np.array([[1, 1], [0, 2]]),
            np.array([[1, 0], [1, 2]]),
            np.array([[0, 0], [0, 2]]),
        ]
        segmenter = DenseMajorityGroundSegmenter.__new__(
            DenseMajorityGroundSegmenter)
        segmenter.backends = [_FakeDenseBackend([value]) for value in maps]
        mask, detections = segmenter.batch(
            [np.zeros((2, 2, 3), np.uint8)])[0]
        np.testing.assert_array_equal(
            mask, np.array([[True, False], [False, True]]))
        by_label = {item.label: item.mask for item in detections}
        np.testing.assert_array_equal(
            by_label["floor"], np.array([[True, False], [False, False]]))
        np.testing.assert_array_equal(
            by_label["stairs"], np.array([[False, False], [False, True]]))

    def test_batches_each_model_once(self):
        first = np.array([[1]], np.int16)
        second = np.array([[0]], np.int16)
        segmenter = DenseMajorityGroundSegmenter.__new__(
            DenseMajorityGroundSegmenter)
        segmenter.backends = [
            _FakeDenseBackend([first, second]),
            _FakeDenseBackend([first, second]),
            _FakeDenseBackend([second, second]),
        ]
        outputs = segmenter.batch([
            np.zeros((1, 1, 3), np.uint8),
            np.zeros((1, 1, 3), np.uint8),
        ])
        self.assertTrue(outputs[0][0][0, 0])
        self.assertFalse(outputs[1][0][0, 0])
        self.assertTrue(all(item.last_batch_size == 2
                            for item in segmenter.backends))


if __name__ == "__main__":
    unittest.main()
