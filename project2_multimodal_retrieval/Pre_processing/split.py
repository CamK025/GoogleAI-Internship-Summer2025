import os
import random
import math
from tqdm import tqdm
import shutil
import json


def split_image_ids(image_list, train_ratio=0.75, val_ratio=0.06, test_ratio=0.19):
    """
    Equals to Localized Narratives total dataset proportion
    """
    random.shuffle(image_list)
    total = len(image_list)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_ids = set(image_list[:train_end])
    val_ids = set(image_list[train_end:val_end])
    test_ids = set(image_list[val_end:])

    print(f"Split: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")
    return {'train': train_ids, 'val': val_ids, 'test': test_ids}

def split_images_to_folders(image_folder, split_ids_dict, output_base_dir):
    """
    Copies images into train_images/ val_images/ test_images/ folders
    according to split_ids_dict.
    """
    for split_name, ids in split_ids_dict.items():
        split_folder = os.path.join(output_base_dir, f'{split_name}_images')
        os.makedirs(split_folder, exist_ok=True)

        for image_id in tqdm(ids, desc=f'Copying {split_name} images'):
            for ext in ['.jpg', '.jpeg', '.png']:
                src = os.path.join(image_folder, f'{image_id}{ext}')
                if os.path.exists(src):
                    dst = os.path.join(split_folder, f'{image_id}{ext}')
                    shutil.copy2(src, dst)
                    break 

def split_jsonl_by_image_ids(filtered_entries, split_ids_dict, output_dir):
    """
    Split filtered_entries into train/val/test JSONL files according to image IDs.
    """
    os.makedirs(output_dir, exist_ok=True)

    splits = {'train': [], 'val': [], 'test': []}

    for entry in filtered_entries:
        image_id = entry['image_id']
        if image_id in split_ids_dict['train']:
            splits['train'].append(entry)
        elif image_id in split_ids_dict['val']:
            splits['val'].append(entry)
        elif image_id in split_ids_dict['test']:
            splits['test'].append(entry)

    for split_name, entries in splits.items():
        output_file = os.path.join(output_dir, f'{split_name}.jsonl')
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        print(f"[{split_name}] Saved {len(entries)} entries to {output_file}.")