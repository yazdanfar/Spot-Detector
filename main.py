import argparse
import cv2
import torch
import os
import matplotlib.pyplot as plt
import tempfile
from blob_detector import BlobDetector


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


def get_default_image(data_dir):
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    for filename in os.listdir(data_dir):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            return os.path.join(data_dir, filename)
    return None


def main():
    parser = argparse.ArgumentParser(description="Detect blobs in an image.")
    parser.add_argument("--image_path", help="Path to the input image")
    parser.add_argument("--model_name", default="best_model.pth",
                        help="Name of the pre-trained model file in the 'models' directory")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device to use for computation")
    parser.add_argument("--output", help="Path to save the output image")
    args = parser.parse_args()

    # Set up paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "models", args.model_name)
    data_dir = os.path.join(current_dir, "data")

    # Use default image if not provided
    if args.image_path is None:
        args.image_path = get_default_image(data_dir)
        if args.image_path is None:
            print("Error: No image found in the data directory.")
            return
        print(f"Using default image: {args.image_path}")

    # Check if the model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return

    # Check if the image file exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image file '{args.image_path}' not found.")
        return

    # Load the pre-trained model
    model = load_model(model_path)

    # Initialize the BlobDetector
    detector = BlobDetector(model, device=args.device)

    # Read the original image
    original_image = cv2.imread(args.image_path)

    # Process the image
    result_image, total_blobs, blobs_r5, blobs_r10 = detector.process_image(args.image_path)

    if result_image is not None:
        # Plot the result
        fig = plot_result(original_image, result_image, total_blobs, blobs_r5, blobs_r10)

        # Save the plot as a temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_filename = temp_file.name
            fig.savefig(temp_filename)
            print(f"Temporary result image saved to {temp_filename}")

        # Save the processed image if output path is provided
        if args.output:
            cv2.imwrite(args.output, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
            print(f"Processed image saved to {args.output}")

        print(f"Total blobs detected: {total_blobs}")
        print(f"Blobs with r<5: {blobs_r5}")
        print(f"Blobs with 5<=r<10: {blobs_r10}")

        # Show the plot
        plt.show()
    else:
        print("No ROI detected in the image.")


if __name__ == "__main__":
    main()