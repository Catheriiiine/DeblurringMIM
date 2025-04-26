import os
import cv2
import shutil
import numpy as np

def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def replace_dir(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

def compute_variance_of_laplacian(image):
    """
    Compute the variance of the Laplacian (VoL) as a measure of image sharpness.
    Lower VoL indicates a blurrier image.
    """
    return cv2.Laplacian(image, cv2.CV_64F).var()

def blur_images_in_subfolder(src_subfolder, dst_subfolder, sigma, image_extensions=('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
    """
    Process all images in src_subfolder:
      - Reads the image in grayscale.
      - Applies a fixed Gaussian blur with the specified sigma.
      - Saves the blurred image in dst_subfolder.
      - Returns a list of VoL values for the processed images.
    """
    vol_list = []
    for filename in os.listdir(src_subfolder):
        if filename.lower().endswith(image_extensions):
            src_path = os.path.join(src_subfolder, filename)
            img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: Could not load {src_path}. Skipping.")
                continue
            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(img, (0, 0), sigma)
            vol = compute_variance_of_laplacian(blurred)
            vol_list.append(vol)
            dst_path = os.path.join(dst_subfolder, filename)
            cv2.imwrite(dst_path, blurred)
            print(f"Saved {dst_path}: VoL = {vol:.2f}")
    return vol_list

def report_statistics(vol_list, label):
    """
    Compute and print summary statistics for a list of VoL values.
    """
    if not vol_list:
        print(f"No images processed for {label}.")
        return
    vol_array = np.array(vol_list)
    print(f"\n--- VoL Statistics for {label} Images ---")
    print(f"Number of images: {len(vol_array)}")
    print(f"Min VoL: {vol_array.min():.2f}")
    print(f"Max VoL: {vol_array.max():.2f}")
    print(f"Mean VoL: {vol_array.mean():.2f}")
    print(f"Median VoL: {np.median(vol_array):.2f}")
    print(f"Std Dev: {vol_array.std():.2f}")

def main():
    # Specify your source folder containing the images to blur.
    src_folder = "/home/catherine/Desktop/DeblurringMIM/leveled_ur_combined"  # Update with your source path.
    
    # Specify the destination folder where blurred images will be saved.
    dst_folder = "/home/catherine/Desktop/DeblurringMIM/leveled_ur_2.5"  # Update with your destination path.
    
    # Prepare the destination folder (clear if exists)
    replace_dir(dst_folder)
    
    # Define the subfolders (labels)
    labels = ["negative", "positive"]

    # Set the sigma for Gaussian blur (adjust to control the level of blur).
    sigma = 2.5
    
    # Process each label folder and gather VoL statistics.
    for label in labels:
        src_subfolder = os.path.join(src_folder, label)
        dst_subfolder = os.path.join(dst_folder, label)
        make_dir(dst_subfolder)
        print(f"\nProcessing {label} images...")
        vol_list = blur_images_in_subfolder(src_subfolder, dst_subfolder, sigma)
        report_statistics(vol_list, label)

if __name__ == '__main__':
    main()
