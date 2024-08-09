import pytest
import numpy as np
import cv2
import torch
from blob_detector import BlobDetector


class MockModel:
    def predict(self, x):
        return torch.rand(1, 1, 480, 320)


@pytest.fixture
def detector():
    model = MockModel()
    return BlobDetector(model)


def test_predict_mask_happy_path(detector):
    image = np.random.rand(480, 320, 3)
    mask = detector.predict_mask(image)
    assert mask.shape == (480, 320)
    assert np.all((mask == 0) | (mask == 1))




def test_detect_blobs_happy_path(detector):
    roi = np.random.rand(100, 100, 3)
    blobs = detector.detect_blobs(roi)
    assert blobs.shape[1] == 3  # x, y, r


def test_get_roi_happy_path(detector):
    image = np.zeros((480, 320, 3), dtype=np.uint8)
    image[100:200, 100:200] = 255
    binary_image = np.zeros((480, 320), dtype=np.uint8)
    binary_image[100:200, 100:200] = 255
    roi, bbox = detector.get_roi(image, binary_image)
    assert roi is not None
    assert bbox is not None




def test_draw_blobs_happy_path(detector):
    roi = np.zeros((100, 100, 3), dtype=np.uint8)
    blobs = np.array([[50, 50, 3], [25, 25, 7], [75, 75, 12]])
    blobs_r5, blobs_r10 = detector.draw_blobs(roi, blobs)
    assert blobs_r5 == 1
    assert blobs_r10 == 1



