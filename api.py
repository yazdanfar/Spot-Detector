import streamlit as st
import sys
import os
import matplotlib
import matplotlib.pyplot as plt
import cv2
import torch
import numpy as np
from PIL import Image
from blob_detector import BlobDetector
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Streamlit page config
st.set_page_config(page_title="Blob Detector App", layout="wide")

# Display system information
st.write(f"Python version: {sys.version}")
st.write(f"Current working directory: {os.getcwd()}")
st.write(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not Set')}")
st.write(f"OpenCV version: {cv2.__version__}")
st.write(f"PyTorch version: {torch.__version__}")
st.write(f"NumPy version: {np.__version__}")
st.write(f"Matplotlib version: {matplotlib.__version__}")
st.write(f"Pillow version: {Image.__version__}")


@st.cache_resource
def load_model(model_path):
    try:
        model = torch.load(model_path, map_location=torch.device('cpu'))
        model.eval()
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        st.error(f"Failed to load model: {e}")
        return None


def plot_result(original_image, result_image, total_blobs, blobs_r5, blobs_r10, density_score):
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        ax1.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        ax1.set_title("Original Image")
        ax1.axis('off')

        ax2.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        ax2.set_title("Processed Image")
        ax2.axis('off')

        plt.suptitle(
            f"Total blobs: {total_blobs} | r<5: {blobs_r5} | 5<=r<10: {blobs_r10} | Density score: {density_score:.2f}")
        plt.tight_layout()

        return fig
    except Exception as e:
        logger.error(f"Error in plot_result: {e}")
        st.error(f"Error in plot_result: {e}")
        return None


def main():
    st.title("Blob Detector Web App")

    # Set up paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "models", "best_model.pth")

    # Check if the model file exists
    if not os.path.exists(model_path):
        logger.error(f"Error: Model file '{model_path}' not found.")
        st.error(f"Error: Model file '{model_path}' not found.")
        return

    # Load the pre-trained model
    model = load_model(model_path)
    if model is None:
        return

    # Initialize the BlobDetector
    try:
        detector = BlobDetector(model, device="cpu")  # Change to "cuda" if using GPU
        logger.info("BlobDetector initialized successfully")
        st.success("BlobDetector initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize BlobDetector: {e}")
        st.error(f"Failed to initialize BlobDetector: {e}")
        return

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])

    if uploaded_file is not None:
        try:
            # Read the image
            image = Image.open(uploaded_file)
            image_np = np.array(image)

            # Convert RGB to BGR (OpenCV uses BGR)
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                original_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            else:
                original_image = image_np  # Grayscale image

            # Process the image
            result = detector.process_image(original_image)

            if result[0] is not None:
                result_image, total_blobs, blobs_r5, blobs_r10, density_score = result

                # Plot the result
                fig = plot_result(original_image, result_image, total_blobs, blobs_r5, blobs_r10, density_score)

                if fig is not None:
                    # Display the plot
                    st.pyplot(fig)

                    # Display blob statistics
                    st.write(f"Total blobs detected: {total_blobs}")
                    st.write(f"Blobs with r<5: {blobs_r5}")
                    st.write(f"Blobs with 5<=r<10: {blobs_r10}")
                    st.write(f"Density score: {density_score:.2f}")

                    # Option to download the result
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png")
                    buf.seek(0)
                    st.download_button(
                        label="Download Result",
                        data=buf,
                        file_name="result.png",
                        mime="image/png"
                    )
            else:
                logger.warning("No ROI detected in the image.")
                st.warning("No ROI detected in the image.")
        except Exception as e:
            logger.error(f"An error occurred while processing the image: {e}")
            st.error(f"An error occurred while processing the image: {e}")
    else:
        st.write("Upload an image to get started!")


if __name__ == "__main__":
    main()