import os
import cv2
import numpy as np

def compute_variance_of_laplacian(image):
    """
    Compute the variance of the Laplacian of the image.
    A higher value indicates a sharper image; lower indicates more blur.
    """
    return cv2.Laplacian(image, cv2.CV_64F).var()

def gather_vol_statistics(folder, extensions=('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
    """
    Walks through the given folder (and subfolders) to process all images
    that match the specified file extensions. Computes and collects the VoL for each image.
    Returns a list of VoL values.
    """
    vol_list = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(extensions):
                path = os.path.join(root, file)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Warning: Failed to load image {path}. Skipping.")
                    continue
                vol = compute_variance_of_laplacian(img)
                vol_list.append(vol)
    return vol_list

def report_statistics(vol_list):
    """
    Computes and prints summary statistics (min, max, mean, median, std)
    for a list of VoL values.
    """
    if not vol_list:
        print("No images were processed.")
        return
    vol_array = np.array(vol_list)
    print("--- VoL Statistics ---")
    print(f"Number of images: {len(vol_array)}")
    print(f"Min VoL: {vol_array.min():.2f}")
    print(f"Max VoL: {vol_array.max():.2f}")
    print(f"Mean VoL: {vol_array.mean():.2f}")
    print(f"Median VoL: {np.median(vol_array):.2f}")
    print(f"Std Dev: {vol_array.std():.2f}")

def main():
    # Specify the folder containing your images.
    folder = "/home/catherine/Desktop/DeblurringMIM/leveled_ur_1.5"  # UPDATE this path as needed
    vol_list = gather_vol_statistics(folder)
    report_statistics(vol_list)

if __name__ == '__main__':
    main()
