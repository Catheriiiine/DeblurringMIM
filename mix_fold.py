import os
import shutil

def combine_train_folds(input_dir, output_dir, folds_to_combine, combined_name="train"):
    """
    Combine specified fold folders (e.g., fold_1, fold_2, fold_3) into one folder called combined_name.
    Assumes each fold folder has subfolders "positive" and "negative".
    """
    combined_folder = os.path.join(output_dir, combined_name)
    pos_combined = os.path.join(combined_folder, "positive")
    neg_combined = os.path.join(combined_folder, "negative")
    os.makedirs(pos_combined, exist_ok=True)
    os.makedirs(neg_combined, exist_ok=True)

    for fold in folds_to_combine:
        fold_name = f"fold_{fold}"
        fold_path = os.path.join(input_dir, fold_name)
        if not os.path.exists(fold_path):
            print(f"Folder {fold_path} does not exist. Skipping fold {fold}.")
            continue

        for label in ["positive", "negative"]:
            src_label_dir = os.path.join(fold_path, label)
            if not os.path.exists(src_label_dir):
                print(f"Folder {src_label_dir} does not exist. Skipping label '{label}' in {fold_name}.")
                continue

            for filename in os.listdir(src_label_dir):
                src_file = os.path.join(src_label_dir, filename)
                # In combined folder, we simply copy the file; optionally you can prefix with the fold number
                dest_file = os.path.join(pos_combined if label=="positive" else neg_combined, filename)
                shutil.copy2(src_file, dest_file)
                # print(f"Copied {src_file} to {dest_file}")

def copy_fold_as_is(input_dir, output_dir, fold):
    """
    Copy a fold folder (e.g., fold_4 or fold_5) from the input directory to the output directory.
    """
    fold_name = f"fold_{fold}"
    src_folder = os.path.join(input_dir, fold_name)
    dest_folder = os.path.join(output_dir, fold_name)
    if not os.path.exists(src_folder):
        print(f"Folder {src_folder} does not exist. Skipping fold {fold}.")
        return
    if os.path.exists(dest_folder):
        print(f"Folder {dest_folder} already exists. Removing it first.")
        shutil.rmtree(dest_folder)
    shutil.copytree(src_folder, dest_folder)
    # print(f"Copied entire folder {src_folder} to {dest_folder}")

if __name__ == "__main__":
    input_folds_dir = "/home/catherine/Desktop/Thickened-Synovium/syn/5_folds_mixed"   # Directory containing fold_1, fold_2, ..., fold_5
    output_base_dir = "/home/catherine/Desktop/Thickened-Synovium/syn/5_folds_mixed_4"      # This is the new output base directory

    # Ensure output directory exists
    os.makedirs(output_base_dir, exist_ok=True)

    # Combine fold_1, fold_2, and fold_3 into one folder called "train"
    combine_train_folds(input_folds_dir, output_base_dir, folds_to_combine=[5,1,2], combined_name="train")

    # Copy fold_4 and fold_5 as they are into the output directory
    for fold in [3, 4]:
        copy_fold_as_is(input_folds_dir, output_base_dir, fold)

    print("Combination complete!")

