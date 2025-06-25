import os
import json
from tqdm import tqdm

def extract_unique_image_ids(jsonl_file):
    image_ids = set()
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line.strip())
            image_id = entry['image_id']
            image_ids.add(image_id)
    return len(image_ids), image_ids

def clean_and_filter_jsonl(base_dir, split_name, image_folder_name, jsonl_name, filtered_json_name):
    """
    Clean and filter a JSONL file:
    - Only keep entries that have matching image files in the image folder.
    - Trim 'traces' to include only points within the time range defined by the first and last timed captions.
    - Remove points whose x or y values are outside the range [0, 1].

    Args:
        base_dir (str): Root directory of the dataset.
        split_name (str): Dataset split (e.g., train/validation/test).
        image_folder_name (str): Name of the image folder.
        jsonl_name (str): Name of the original JSONL file.
        filtered_json_name (str): Name of the filtered JSONL file.

    Returns:
        list: A list of filtered JSONL entries.
    """

    # Define file paths
    jsonl_file = os.path.join(base_dir, jsonl_name)
    image_folder = os.path.join(base_dir, image_folder_name)
    filtered_json_file = os.path.join(base_dir, filtered_json_name)

    # Collect all valid image filenames
    image_files = os.listdir(image_folder)
    image_list = [img for img in image_files if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"[{split_name}] Found {len(image_list)} images.")
    # print(f"[{split_name}] Found {len(image_list)} images in {image_folder} folder.")

    # Load existing filtered file if available
    if os.path.exists(filtered_json_file):
        with open(filtered_json_file, 'r', encoding='utf-8') as f:
            filtered_entries = json.load(f)
    else:
        entries = {}
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"[{split_name}] Filtering JSONL and trimming traces"):
                entry = json.loads(line.strip())
                image_id = entry['image_id']

                # Check if the image exists in the folder
                if (f"{image_id}.jpg" in image_list or
                    f"{image_id}.jpeg" in image_list or
                    f"{image_id}.png" in image_list):

                    if image_id not in entries:
                        timed_caption = entry.get('timed_caption', [])
                        if timed_caption:
                            first_start = timed_caption[0]['start_time']
                            last_end = timed_caption[-1]['end_time']
                            original_traces = entry.get('traces', [])
                            trimmed_traces = []

                            for trace in original_traces:
                                # Keep only points within the timed_caption range
                                filtered_points = [
                                    pt for pt in trace
                                    if first_start <= pt['t'] <= last_end
                                    and 0.0 <= pt['x'] <= 1.0
                                    and 0.0 <= pt['y'] <= 1.0
                                ]
                                if filtered_points:
                                    trimmed_traces.append(filtered_points)
                            entry['traces'] = trimmed_traces
                        else:
                            entry['traces'] = []

                        entries[image_id] = entry

        # Convert dictionary to list
        filtered_entries = list(entries.values())

        # Save the filtered entries to a new JSON file
        with open(filtered_json_file, 'w', encoding='utf-8') as f:
            for entry in filtered_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"[{split_name}] Number of filtered entries: {len(filtered_entries)}")
    return filtered_entries

