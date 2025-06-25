import os
import pandas as pd
import requests
from tqdm import tqdm
from .clean import extract_unique_image_ids

def download_matched_images(jsonl_file, csv_file, output_dir):

    _, unique_image_ids = extract_unique_image_ids(jsonl_file)

    df = pd.read_csv(csv_file)
    df_matched = df[df['ImageID'].isin(unique_image_ids)]

    os.makedirs(output_dir, exist_ok=True)

    failed_downloads = []
    for idx, row in tqdm(df_matched.iterrows(), total=len(df_matched)):
        image_id = row['ImageID']
        url = row['OriginalURL']
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(os.path.join(output_dir, f"{image_id}.jpg"), 'wb') as f:
                    f.write(response.content)
            else:
                print(f"[WARNING] Failed to download {image_id}: HTTP {response.status_code}")
                failed_downloads.append((image_id, url))
        except Exception as e:
            # print(f"[ERROR] Failed to download {image_id}: {e}")
            failed_downloads.append((image_id, url))

    print("Download completed.")
    return failed_downloads

