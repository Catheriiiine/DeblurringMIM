import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_and_display_laplacian(image_path):
    """
    Reads an image, computes its Laplacian, converts the result to an 8-bit image,
    and displays both the original and the Laplacian image using Matplotlib.
    """
    # Read the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not open or find the image {image_path}")

    # Compute the Laplacian
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    laplacian_abs = cv2.convertScaleAbs(laplacian)
    
    # Use Matplotlib to display the images
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title("Original Image")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(laplacian_abs, cmap='gray')
    plt.title("Laplacian Image")
    plt.axis("off")
    
    plt.tight_layout()
    plt.savefig("/home/catherine/Desktop/DeblurringMIM/vol_image6.png")
    plt.show()

# Example usage:
if __name__ == "__main__":
    image_path = "/home/catherine/Desktop/Thickened-Synovium/ur_cropped/top/negative/0083_02__KR__30_06_21__16_41_53.png"  
    get_and_display_laplacian(image_path)
