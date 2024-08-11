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


# ... (previous code remains the same)

def main():
    # ... (previous code remains the same)

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