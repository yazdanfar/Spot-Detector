from flask import Flask, request, jsonify
import torch
import cv2
import numpy as np
import os
from blob_detector import BlobDetector
from PIL import Image
import io

app = Flask(__name__)

def load_model(model_path):
    # This is a placeholder function. You'll need to implement
    # the actual model loading logic based on your model architecture.
    model = torch.load(model_path)
    model.eval()
    return model

# Load the model at startup
model_name = "best_model.pth"
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "models", model_name)
model = load_model(model_path)

# Initialize the BlobDetector
detector = BlobDetector(model, device="cpu")  # Change to "cuda" if using GPU

@app.route('/detect_blobs', methods=['POST'])
def detect_blobs():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    image = Image.open(image_file)
    image_np = np.array(image)
    original_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    result_image, total_blobs, blobs_r5, blobs_r10 = detector.process_image(original_image)

    if result_image is not None:
        # Convert result image to base64 for JSON response
        _, buffer = cv2.imencode('.png', result_image)
        result_image_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'total_blobs': int(total_blobs),
            'blobs_r5': int(blobs_r5),
            'blobs_r10': int(blobs_r10),
            'result_image': result_image_base64
        })
    else:
        return jsonify({'error': 'No ROI detected in the image'}), 400

if __name__ == '__main__':
    app.run(debug=True)