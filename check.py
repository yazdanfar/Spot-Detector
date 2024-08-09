from blob_detector import BlobDetector
import matplotlib.pyplot as plt
import torch
import logging

print('hi', torch.__version__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load your pre-trained model here
model = torch.load('models/best_model.pth')

detector = BlobDetector(model)
result_image, total_blobs, blobs_r5, blobs_r10 = detector.process_image('new.jpg')

if result_image is not None:
    print(f"Total blobs: {total_blobs}")
    print(f"Blobs with r<5: {blobs_r5}")
    print(f"Blobs with 5<=r<10: {blobs_r10}")
    plt.imshow(result_image)
    plt.show()

else:
    print("No ROI detected in the image")


