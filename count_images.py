import os

def count_files_two_levels(root_dir):
    """
    For each directory in root_dir, look inside it for subfolders.
    In each subfolder, count and print the number of files.
    """
    # First, list all top-level directories in root_dir
    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        
        # Ensure this top-level item is indeed a directory
        if os.path.isdir(folder_path):
            # Now, look at each subfolder inside this directory
            for subfolder in os.listdir(folder_path):
                subfolder_path = os.path.join(folder_path, subfolder)
                
                # Check if this is indeed a subfolder
                if os.path.isdir(subfolder_path):
                    # Count files in subfolder (ignoring further directories)
                    files = [
                        f for f in os.listdir(subfolder_path)
                        if os.path.isfile(os.path.join(subfolder_path, f))
                    ]
                    print(f"{folder}/{subfolder}: {len(files)} file(s)")

if __name__ == "__main__":
    # Replace this path with the actual path of your top-level directory
    root_directory = "/home/catherine/Desktop/DeblurringMIM/blurred_fold_3"
    # root_directory = "/home/catherine/Desktop/Thickened-Synovium/syn/mixed_images_final"
    count_files_two_levels(root_directory)
