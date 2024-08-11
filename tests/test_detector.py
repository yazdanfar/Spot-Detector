import pytest
import numpy as np
import cv2
import torch
from blob_detector import BlobDetector
import os

# Adjust the path to import from the parent directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture
def detector():
    # Adjust the path to load the model from the correct directory
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'best_model.pth')
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()  # Set the model to evaluation mode
    return BlobDetector(model)

def test_predict_mask_happy_path(detector):
    image = np.random.rand(480, 320, 3).astype(np.float32)
    mask = detector.predict_mask(image)
    assert mask.shape == (480, 320)
    assert np.all((mask >= 0) & (mask <= 1))  # Allow for float values between 0 and 1

def test_detect_blobs_happy_path(detector):
    roi = np.random.rand(100, 100, 3)
    blobs = detector.detect_blobs(roi)
    assert blobs.shape[1] == 3  # y, x, r

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
    roi_area = 10000  # 100 * 100
    result = detector.draw_blobs(roi, blobs, roi_area)
    assert len(result) == 3
    assert all(isinstance(x, (int, float)) for x in result)

def test_calculate_density_score(detector):
    blobs = np.array([[50, 50, 3], [25, 25, 7], [75, 75, 12]])
    roi_area = 10000  # 100 * 100
    score = detector.calculate_density_score(blobs, roi_area)
    assert isinstance(score, float)
    assert score >= 0

def test_process_binary_image(detector):
    binary_image = np.zeros((100, 100), dtype=np.uint8)
    binary_image[40:60, 40:60] = 1
    processed = detector.process_binary_image(binary_image)
    assert processed.shape == (100, 100)
    assert not np.array_equal(processed, binary_image), "Processed image should be different from input"

def test_preprocess_image(detector, tmp_path):
    image_path = tmp_path / "test_image.png"
    image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(image_path), image)

    result = detector.preprocess_image(str(image_path))
    assert len(result) == 3
    assert all(isinstance(img, np.ndarray) for img in result)
    assert result[1].shape[:2] == (480, 320)  # Check only height and width of resized image
    assert result[2].shape[:2] == (480, 320)  # Check only height and width of RGB image

def test_process_image(detector, tmp_path):
    image_path = tmp_path / "test_image.png"
    image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(image_path), image)

    result = detector.process_image(str(image_path))
    assert len(result) == 5
    assert isinstance(result[0], (np.ndarray, type(None)))
    assert all(isinstance(x, (int, float)) for x in result[1:])