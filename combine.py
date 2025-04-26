import os
import shutil

def copy_images(source_dir, dest_dir):
    # Walk through every directory in the source folder
    for root, dirs, files in os.walk(source_dir):
        # Check if the current folder's name is "negative" or "positive"
        folder_name = os.path.basename(root)
        if folder_name in ("negative", "positive"):
            # Define destination subfolder (e.g., new/negative or new/positive)
            dest_subfolder = os.path.join(dest_dir, folder_name)
            os.makedirs(dest_subfolder, exist_ok=True)
            
            for file in files:
                # Filter image files (adjust extensions if needed)
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    source_file = os.path.join(root, file)
                    dest_file = os.path.join(dest_subfolder, file)
                    
                    # Optionally handle duplicate filenames here
                    shutil.copy2(source_file, dest_file)
                    print(f"Copied: {source_file} -> {dest_file}")

if __name__ == '__main__':
    # Specify your source and destination directories here:
    source_directory = "/home/catherine/Desktop/DeblurringMIM/leveled_ur"           # Change to your source folder path
    destination_directory = "/home/catherine/Desktop/DeblurringMIM/leveled_ur_combined"   # Change to your destination folder path
    
    copy_images(source_directory, destination_directory)
