import cv2
import numpy as np
import torch
from skimage import color
from skimage.util import img_as_float
from skimage.feature import blob_log
from skimage import draw
from typing import Tuple, Optional
import logging
import time
import psutil
import os
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BlobDetector:
    """A class for detecting blobs in images using a pre-trained model."""

    def __init__(self, model: torch.nn.Module, device: str = 'cpu'):
        """
        Initialize the BlobDetector.

        Args:
            model (torch.nn.Module): The pre-trained model for mask prediction.
            device (str): The device to use for computations ('cpu' or 'cuda').
        """
        self.model = model
        self.device = device

    def preprocess_image(self, image_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess the input image.

        Args:
            image_path (str): Path to the input image.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Original image, resized image, and RGB image.
        """
        original_image = cv2.imread(image_path)
        resized_image = cv2.resize(original_image, (320, 480))
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        return original_image, resized_image, rgb_image

    def predict_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Predict the mask for the input image.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Predicted binary mask.
        """
        x_tensor = torch.from_numpy(image).float().to(self.device)
        x_tensor /= 255.0
        x_tensor = x_tensor.permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            pr_mask = self.model(x_tensor)
        return (pr_mask.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

    def process_binary_image(self, binary_image: np.ndarray) -> np.ndarray:
        """
        Process the binary image with erosion and dilation.

        Args:
            binary_image (np.ndarray): Input binary image.

        Returns:
            np.ndarray: Processed binary image.
        """
        binary_image *= 255
        kernel = np.ones((9, 9), np.uint8)
        eroded = cv2.erode(binary_image, kernel, anchor=(0, 0), iterations=3)
        dilated = cv2.dilate(eroded, kernel, anchor=(0, 0), iterations=20)
        return dilated

    def get_roi(self, image: np.ndarray, binary_image: np.ndarray) -> Tuple[
        Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        """
        Get the region of interest (ROI) from the image.

        Args:
            image (np.ndarray): Input image.
            binary_image (np.ndarray): Binary image for ROI detection.

        Returns:
            Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]: ROI image and bounding box coordinates.
        """
        mask = cv2.inRange(binary_image, 127, 255)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            roi = image[y:y+h, x:x+w]
            return roi, (x, y, w, h)
        return None, None

    def detect_blobs(self, roi: np.ndarray) -> np.ndarray:
        """
        Detect blobs in the ROI.

        Args:
            roi (np.ndarray): Region of interest image.

        Returns:
            np.ndarray: Detected blobs.
        """
        roi_gray = img_as_float(color.rgb2gray(roi))
        blobs_log = blob_log(roi_gray, min_sigma=1, max_sigma=20, num_sigma=100, threshold=.05)
        blobs_log[:, 2] = blobs_log[:, 2] * np.sqrt(2)
        return blobs_log

    def calculate_density_score(self, blobs: np.ndarray, roi_area: float) -> float:
        """
        Calculate a density-based logarithmic score.

        Args:
            blobs (np.ndarray): Detected blobs.
            roi_area (float): Area of the region of interest in pixels.

        Returns:
            float: Density-based logarithmic score.
        """
        spot_count = len(blobs)
        if spot_count == 0 or roi_area == 0:
            return 0.0

        density = spot_count / roi_area
        return math.log10(1 + density * 1000)  # Multiply by 1000 to scale up small densities

    def draw_blobs(self, roi: np.ndarray, blobs: np.ndarray, roi_area: float) -> Tuple[int, int, float]:
        """
        Draw detected blobs on the ROI image and calculate the density-based score.

        Args:
            roi (np.ndarray): Region of interest image.
            blobs (np.ndarray): Detected blobs.
            roi_area (float): Area of the region of interest in pixels.

        Returns:
            Tuple[int, int, float]: Number of blobs with r<5, 5<=r<10, and the density-based score.
        """
        blobs_r5 = 0
        blobs_r10 = 0
        density_score = self.calculate_density_score(blobs, roi_area)

        for blob in blobs:
            y, x, r = blob
            if r < 5:
                blobs_r5 += 1
                color = [255, 0, 0]  # Red for r < 5
            elif r < 10:
                blobs_r10 += 1
                color = [0, 255, 0]  # Green for 5 <= r < 10
            else:
                color = [0, 0, 255]  # Blue for r >= 10
            rr, cc = draw.circle_perimeter(int(y), int(x), int(r), shape=roi.shape[:2])
            roi[rr, cc] = color
        return blobs_r5, blobs_r10, density_score

    def process_image(self, image_path: str) -> Tuple[Optional[np.ndarray], int, int, int, float]:
        """
        Process an image for blob detection and score calculation.

        Args:
            image_path (str): Path to the input image.

        Returns:
            Tuple[Optional[np.ndarray], int, int, int, float]:
                Processed image with blobs detected, total blobs, blobs with r<5, blobs with 5<=r<10, and density-based score.
                If no ROI is detected, returns (None, 0, 0, 0, 0.0).
        """
        start_time = time.time()
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / 1024 / 1024  # Memory in MB

        logger.info(f"Starting image processing for {image_path}")

        original, resized, rgb = self.preprocess_image(image_path)
        logger.info("Image preprocessed")

        mask = self.predict_mask(rgb)
        logger.info("Mask predicted")

        processed_mask = self.process_binary_image(mask)
        logger.info("Binary image processed")

        roi, bbox = self.get_roi(rgb.copy(), processed_mask)
        if roi is not None:
            logger.info(f"ROI detected: {bbox}")

            blobs = self.detect_blobs(roi)
            logger.info(f"Blobs detected: {len(blobs)}")

            roi_area = bbox[2] * bbox[3]  # width * height
            blobs_r5, blobs_r10, density_score = self.draw_blobs(roi, blobs, roi_area)
            logger.info(f"Blobs drawn: {blobs_r5} with r<5, {blobs_r10} with 5<=r<10")
            logger.info(f"Density-based score: {density_score:.2f}")

            # Put text on the image
            font = cv2.FONT_HERSHEY_PLAIN
            cv2.putText(rgb, f'Total blobs: {len(blobs)}', (10, 420), font, 1, (100, 150, 0), 1, cv2.LINE_AA)
            cv2.putText(rgb, f'Blobs r<5: {blobs_r5}', (10, 435), font, 1, (100, 150, 0), 1, cv2.LINE_AA)
            cv2.putText(rgb, f'Blobs 5<=r<10: {blobs_r10}', (10, 450), font, 1, (100, 150, 0), 1, cv2.LINE_AA)
            cv2.putText(rgb, f'Density score: {density_score:.2f}', (10, 465), font, 1, (100, 150, 0), 1, cv2.LINE_AA)

            # Draw rectangle and blobs on the original image
            x, y, w, h = bbox
            cv2.rectangle(rgb, (x, y), (x + w, y + h), (255, 0, 0), 10)
            rgb[y:y + h, x:x + w] = cv2.addWeighted(rgb[y:y + h, x:x + w], 0.5, roi, 0.5, 0)

            end_time = time.time()
            end_memory = process.memory_info().rss / 1024 / 1024  # Memory in MB

            logger.info(f"Image processing completed in {end_time - start_time:.2f} seconds")
            logger.info(f"Memory usage: {end_memory - start_memory:.2f} MB")

            return rgb, len(blobs), blobs_r5, blobs_r10, density_score
        else:
            logger.warning("No ROI detected")
            end_time = time.time()
            end_memory = process.memory_info().rss / 1024 / 1024  # Memory in MB

            logger.info(f"Image processing completed in {end_time - start_time:.2f} seconds")
            logger.info(f"Memory usage: {end_memory - start_memory:.2f} MB")

            return None, 0, 0, 0, 0.0