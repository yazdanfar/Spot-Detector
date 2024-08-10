import streamlit as st
import torch
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from blob_detector import BlobDetector
from io import BytesIO
from PIL import Image


@st.cache_resource
def load_model(model_path):
    # This is a placeholder function. You'll need to implement
    # the actual model loading logic based on your model architecture.
    model = torch.load(model_path)
    model.eval()
    return model


def plot_result(original_image, result_image, total_blobs, blobs_r5, blobs_r10):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    ax1.set_title("Original Image")
    ax1.axis('off')

    ax2.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    ax2.set_title("Processed Image")
    ax2.axis('off')

    plt.suptitle(f"Total blobs: {total_blobs} | r<5: {blobs_r5} | 5<=r<10: {blobs_r10}")
    plt.tight_layout()

    return fig


def main():
    st.title("Blob Detector Web App")

    # Set the model path
    model_name = "best_model.pth"
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "models", model_name)

    # Check if the model file exists
    if not os.path.exists(model_path):
        st.error(f"Error: Model file '{model_path}' not found.")
        return

    # Load the pre-trained model
    model = load_model(model_path)

    # Initialize the BlobDetector
    detector = BlobDetector(model, device="cpu")  # Change to "cuda" if using GPU

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        # Convert image to BGR (OpenCV format)
        original_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Process the image
        result_image, total_blobs, blobs_r5, blobs_r10 = detector.process_image(original_image)

        if result_image is not None:
            # Plot the result
            fig = plot_result(original_image, result_image, total_blobs, blobs_r5, blobs_r10)

            # Display the plot
            st.pyplot(fig)

            # Display blob statistics
            st.write(f"Total blobs detected: {total_blobs}")
            st.write(f"Blobs with r<5: {blobs_r5}")
            st.write(f"Blobs with 5<=r<10: {blobs_r10}")

            # Option to download the result
            buf = BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            st.download_button(
                label="Download Result",
                data=buf,
                file_name="result.png",
                mime="image/png"
            )
        else:
            st.write("No ROI detected in the image.")


if __name__ == "__main__":
    main()