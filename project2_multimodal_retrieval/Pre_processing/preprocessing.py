import os
import random
import math
from tqdm import tqdm
from .download import download_matched_images
from .clean import clean_and_filter_jsonl
from .split import split_image_ids, split_images_to_folders, split_jsonl_by_image_ids

def main():
    base_dir = './Dataset'
    train_jsonl = 'open_images_train_v6_localized_narratives-00000-of-00010.jsonl'
    train_image_dir = os.path.join(base_dir, 'Original_train')
    output_jsonl_dir = os.path.join(base_dir, 'split_jsonl')
    output_image_base_dir = base_dir  

    # 1. Download images from the train set
    # print("Downloading train images...")
    # download_matched_images(os.path.join(base_dir, train_jsonl), 
    #                         os.path.join(base_dir, train_csv), 
    #                         train_image_dir)

    # 2. Divide the data proportionally
    image_files = os.listdir(train_image_dir)
    image_ids = [os.path.splitext(img)[0] for img in image_files if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
    split_ids_dict = split_image_ids(image_ids, train_ratio=0.75, val_ratio=0.06, test_ratio=0.19)
    split_images_to_folders(train_image_dir, split_ids_dict, output_image_base_dir)

    # 3. Cleaning, dividing and saving jsonl
    filtered_entries = clean_and_filter_jsonl(base_dir, 'Original Train', 'Original_train', train_jsonl, 'filtered_original_train.jsonl')
    split_jsonl_by_image_ids(filtered_entries, split_ids_dict, output_jsonl_dir)

    print("Preprocessing completed.")

if __name__ == '__main__':
    main()
