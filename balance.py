import os
import random
import shutil

def get_image_files(folder, valid_exts={'.jpg', '.jpeg', '.png', '.bmp'}):
    return [os.path.join(folder, f) for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in valid_exts]

def balance_split(source_split, output_split):
    for cls in ['positive', 'negative']:
        src_folder = os.path.join(source_split, cls)
        files = get_image_files(src_folder)
        print(f"Found {len(files)} {cls} images in {src_folder}")
        # Store the file list for each class
        if cls == 'positive':
            pos_files = files
        else:
            neg_files = files

    # Determine the minimum number of images between classes
    num_to_keep = min(len(pos_files), len(neg_files))
    print(f"Keeping {num_to_keep} images per class in split {os.path.basename(source_split)}")

    # Randomly sample without replacement
    pos_sample = random.sample(pos_files, num_to_keep)
    neg_sample = random.sample(neg_files, num_to_keep)

    # Create output folders if they don't exist
    for cls in ['positive', 'negative']:
        os.makedirs(os.path.join(output_split, cls), exist_ok=True)

    # Copy the files for positive class
    for file_path in pos_sample:
        filename = os.path.basename(file_path)
        dest_path = os.path.join(output_split, 'positive', filename)
        shutil.copy2(file_path, dest_path)

    # Copy the files for negative class
    for file_path in neg_sample:
        filename = os.path.basename(file_path)
        dest_path = os.path.join(output_split, 'negative', filename)
        shutil.copy2(file_path, dest_path)

def create_balanced_dataset(source_dir, output_dir):
    random.seed(123)  # For reproducibility
    splits = ['train', 'val', 'test']
    for split in splits:
        source_split = os.path.join(source_dir, split)
        output_split = os.path.join(output_dir, split)
        os.makedirs(output_split, exist_ok=True)
        print(f"\nProcessing split: {split}")
        balance_split(source_split, output_split)
    print("\nBalanced dataset created successfully!")

if __name__ == '__main__':
    # Update these paths with your actual dataset directories
    source_dataset_dir = '/home/catherine/Desktop/DeblurringMIM/blurred_images_fold_5'
    output_dataset_dir = '/home/catherine/Desktop/DeblurringMIM/classification_images_fold_5'

    create_balanced_dataset(source_dataset_dir, output_dataset_dir)
