# import os
# import cv2
# import numpy as np
# import statistics
# import shutil

# def measure_blur_laplacian(img_path):
#     """
#     Returns the Variance of Laplacian (VoL) for the given image file.
#     A higher VoL typically indicates a sharper image, lower VoL indicates more blur.
#     """
#     img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#     if img is None:
#         # Return None or some sentinel value if file cannot be read as an image.
#         return None
    
#     laplacian = cv2.Laplacian(img, cv2.CV_64F)
#     variance = laplacian.var()
#     return variance

# def measure_blur_in_folder(folder_path):
#     """
#     Recursively goes through 'folder_path' to find image files,
#     calculates the Variance of Laplacian for each, and returns:
#     {file_path: vol_value, ...}
#     """
#     valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
#     results = {}

#     for root, dirs, files in os.walk(folder_path):
#         for file_name in files:
#             if file_name.lower().endswith(valid_extensions):
#                 file_path = os.path.join(root, file_name)
#                 vol = measure_blur_laplacian(file_path)
#                 if vol is not None:
#                     results[file_path] = vol
#                 else:
#                     print(f"Warning: Could not process file: {file_path}")
#     return results

# # def print_statistics(vol_values):
# #     """
# #     Given a list of Vol values (floats), print various statistics including
# #     min, max, and the 25th, 50th, 75th percentiles.
# #     """
# #     if not vol_values:
# #         print("No VoL data available to compute statistics.")
# #         return

# #     # Convert to NumPy array for easy quantile calculations
# #     vol_array = np.array(vol_values)

# #     # Calculate the range and quantiles
# #     vol_min = np.min(vol_array)
# #     vol_max = np.max(vol_array)
# #     # Quantiles at 1/3 and 2/3
# #     q33 = np.quantile(vol_array, 1/3)
# #     q66 = np.quantile(vol_array, 2/3)

# #     # Count images according to VoL thresholds
# #     count_under_50 = np.sum(vol_array < 50)
# #     count_50_200 = np.sum((vol_array >= 50) & (vol_array <= 200))
# #     count_200_400 = np.sum((vol_array >= 200) & (vol_array <= 400))
# #     count_400_1000 = np.sum((vol_array >= 200) & (vol_array <= 1000))
# #     count_over_1000 = np.sum(vol_array > 1000)

# #     print(f"\n--- Variance of Laplacian Statistics ---")
# #     print(f"Min:                 {vol_min:.2f}")
# #     print(f"1/3 Quantile (33%):  {q33:.2f}")
# #     print(f"2/3 Quantile (66%):  {q66:.2f}")
# #     print(f"Max:                 {vol_max:.2f}")

# #     print("\n--- Count by Thresholds ---")
# #     print(f"Number of images with VoL < 50:        {count_under_50}")
# #     print(f"Number of images with 50 <= VoL <= 200: {count_50_200}")
# #     print(f"Number of images with 200 <= VoL <= 400: {count_200_400}")
# #     print(f"Number of images with 400 <= VoL <= 1000: {count_400_1000}")
# #     print(f"Number of images with VoL > 1000:        {count_over_1000}")

# def print_statistics(vol_dict):
#     """
#     Given a dictionary of {file_path: VoL}, print various statistics including
#     the number of images in different VoL ranges and print 5 sample filenames from each range.
#     """
#     if not vol_dict:
#         print("No VoL data available to compute statistics.")
#         return

#     # Convert dictionary to list of tuples sorted by VoL values
#     sorted_vol = sorted(vol_dict.items(), key=lambda x: x[1])

#     # Separate data into defined ranges
#     under_50 = [(path, vol) for path, vol in sorted_vol if vol < 100]
#     between_50_200 = [(path, vol) for path, vol in sorted_vol if 100 < vol < 200]
#     over_1000 = [(path, vol) for path, vol in sorted_vol if vol > 200]

#     # Print statistics
#     print("\n--- Count by Thresholds ---")
#     print(f"Number of images with VoL < 50:        {len(under_50)}")
#     print(f"Number of images with 50 <= VoL <= 200: {len(between_50_200)}")
#     print(f"Number of images with VoL > 1000:       {len(over_1000)}")

#     # Function to print up to 5 samples from each range
#     def print_sample_images(label, images):
#         print(f"\nSample images {label}:")
#         for path, vol in images[500:505]:  # Print at most 5 images
#             print(f"  {path}: VoL = {vol:.2f}")

#     # Print sample images
#     print_sample_images("with VoL < 50", under_50)
#     print_sample_images("with 50 <= VoL <= 200", between_50_200)
#     print_sample_images("with VoL > 1000", over_1000)
# def save_images_by_vol(vol_dict, output_dir):
#     """
#     Saves images into separate directories based on their VoL ranges.
#     """
#     if not vol_dict:
#         print("No VoL data available to categorize images.")
#         return

#     # Define output directories for each range
#     categories = {
#         "VoL_less_100": (lambda vol: vol < 100),
#         "VoL_100_200": (lambda vol: 100 <= vol <= 200),
#         "VoL_over_1000": (lambda vol: vol > 200),
#     }

#     # Create directories if they do not exist
#     for category in categories.keys():
#         os.makedirs(os.path.join(output_dir, category, "positive"), exist_ok=True)

#     # Categorize and move images
#     for img_path, vol in vol_dict.items():
#         for category, condition in categories.items():
#             if condition(vol):
#                 dest_dir = os.path.join(output_dir, category, "positive")
#                 dest_path = os.path.join(dest_dir, os.path.basename(img_path))
#                 shutil.copy(img_path, dest_path)
#                 print(f"Copied {img_path} to {dest_path}")
#                 break  # Once assigned to a category, break to avoid multiple moves


# if __name__ == "__main__":
#     folder_to_test = "/home/catherine/Desktop/Thickened-Synovium/upper_left"
#     output_directory = "/home/catherine/Desktop/DeblurringMIM/leveled_test_ul"
#     blur_results = measure_blur_in_folder(folder_to_test)

#     # vol_values = list(blur_results.values())
#     # print_statistics(vol_values)
    
#     # # Save images into categorized folders
#     # save_images_by_vol(blur_results, output_directory)
#     # Print statistics and sample images
#     print_statistics(blur_results)


import os
import cv2
import numpy as np
import statistics
import shutil
import csv
import matplotlib.pyplot as plt

def measure_blur_laplacian(img_path):
    """
    Returns the Variance of Laplacian (VoL) for the given image file.
    A higher VoL typically indicates a sharper image, lower VoL indicates more blur.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        # Return None or some sentinel value if file cannot be read as an image.
        return None
    
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    variance = laplacian.var()
    return variance

def measure_blur_in_folder(folder_path):
    """
    Recursively goes through 'folder_path' to find image files,
    calculates the Variance of Laplacian for each, and returns:
    {file_path: vol_value, ...}
    """
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    results = {}

    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if file_name.lower().endswith(valid_extensions):
                file_path = os.path.join(root, file_name)
                vol = measure_blur_laplacian(file_path)
                if vol is not None:
                    results[file_path] = vol
                else:
                    print(f"Warning: Could not process file: {file_path}")
    return results

def print_statistics(vol_dict):
    """
    Given a dictionary of {file_path: VoL}, print various statistics including
    the number of images in different VoL ranges and print 5 sample filenames from each range.
    """
    if not vol_dict:
        print("No VoL data available to compute statistics.")
        return

    # Convert dictionary to list of tuples sorted by VoL values
    sorted_vol = sorted(vol_dict.items(), key=lambda x: x[1])

    # Separate data into defined ranges
    under_100 = [(path, vol) for path, vol in sorted_vol if vol < 100]
    between_100_200 = [(path, vol) for path, vol in sorted_vol if 100 <= vol <= 200]
    over_200 = [(path, vol) for path, vol in sorted_vol if vol > 200]

    # Print statistics
    print("\n--- Count by Thresholds ---")
    print(f"Number of images with VoL < 100:        {len(under_100)}")
    print(f"Number of images with 100 <= VoL <= 200: {len(between_100_200)}")
    print(f"Number of images with VoL > 200:        {len(over_200)}")

    # Function to print up to 5 samples from each range
    def print_sample_images(label, images):
        print(f"\nSample images {label}:")
        for path, vol in images[500:505]:  # Print at most 5 images
            print(f"  {path}: VoL = {vol:.2f}")

    # Print sample images
    print_sample_images("with VoL < 100", under_100)
    print_sample_images("with 100 <= VoL <= 200", between_100_200)
    print_sample_images("with VoL > 200", over_200)

def save_vol_to_csv(vol_dict, csv_file_path):
    """
    Saves the VoL results to a CSV file.

    Parameters:
        vol_dict (dict): Dictionary where key is the file path and value is the VoL.
        csv_file_path (str): Path to the CSV file to be created.
    """
    with open(csv_file_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # Write header row
        writer.writerow(["file_path", "vol_value"])
        # Write each file's VoL value
        for file_path, vol in vol_dict.items():
            writer.writerow([file_path, vol])
    print(f"Saved VoL results to {csv_file_path}")

def save_images_by_vol(vol_dict, output_dir):
    """
    Saves images into separate directories based on their VoL ranges.
    """
    if not vol_dict:
        print("No VoL data available to categorize images.")
        return

    # Define output directories for each range
    categories = {
        "VoL_less_100": (lambda vol: vol < 100),
        "VoL_100_200": (lambda vol: 100 <= vol <= 200),
        "VoL_over_200": (lambda vol: vol > 200),
    }

    # Create directories if they do not exist
    for category in categories.keys():
        os.makedirs(os.path.join(output_dir, category, "positive"), exist_ok=True)

    # Categorize and move images
    for img_path, vol in vol_dict.items():
        for category, condition in categories.items():
            if condition(vol):
                dest_dir = os.path.join(output_dir, category, "positive")
                dest_path = os.path.join(dest_dir, os.path.basename(img_path))
                shutil.copy(img_path, dest_path)
                print(f"Copied {img_path} to {dest_path}")
                break  # Once assigned to a category, break to avoid multiple moves


def plot_vol_histogram(vol_dict):
    """
    Plots a histogram of VoL values with bins of width 10.
    
    Parameters:
        vol_dict (dict): Dictionary with image file paths as keys and VoL values as values.
    """
    # Extract VoL values from the dictionary
    vol_values = list(vol_dict.values())
    
    # Determine the range of the data
    min_vol = int(np.floor(min(vol_values)))
    max_vol = int(np.ceil(max(vol_values)))
    
    # Create bins with an interval of 10
    bins = np.arange(min_vol, max_vol + 10, 10)
    
    plt.figure(figsize=(10, 6))
    plt.hist(vol_values, bins=bins, edgecolor='black')
    plt.title("Histogram of Variance of Laplacian (VoL)")
    plt.xlabel("VoL value")
    plt.ylabel("Number of images")
    plt.grid(True)
    plt.savefig("/home/catherine/Desktop/DeblurringMIM/leveled_test_ul/vol_histogram_test.png")
    plt.show()

if __name__ == "__main__":
    folder_to_test = "/home/catherine/Desktop/DeblurringMIM/leveled_ur_combined"
    # output_directory = "/home/catherine/Desktop/DeblurringMIM/leveled_test_ul"
    
    # Measure the VoL for images in the folder
    blur_results = measure_blur_in_folder(folder_to_test)

    # plot_vol_histogram(blur_results)

    # Print statistics and sample images
    print_statistics(blur_results)
    
    # # Save the VoL results to a CSV file
    # csv_output_path = "/home/catherine/Desktop/DeblurringMIM/leveled_test_ul/VoL.csv"
    # os.makedirs(output_directory, exist_ok=True)
    # save_vol_to_csv(blur_results, csv_output_path)
    
    # Optionally, save images into categorized folders based on VoL thresholds
    # save_images_by_vol(blur_results, output_directory)

