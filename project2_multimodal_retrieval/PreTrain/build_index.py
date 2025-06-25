from pathlib import Path
import pickle, numpy as np
from annoy import AnnoyIndex
from tqdm import tqdm
from PIL import Image

from configs.default import JSONL_TRAIN, IMAGES_DIR, INDEX_DIR
from dataset_ln import LNIterableDataset
from model_clip_ln import CLIPEncoder
from utils_bbox import crop_bbox

def build_annoy(max_items=None, n_trees=50):
    out=Path(INDEX_DIR); out.mkdir(parents=True, exist_ok=True)
    ds  = LNIterableDataset(JSONL_TRAIN, IMAGES_DIR, max_items=max_items)
    enc = CLIPEncoder()
    feats, meta = [], []
    for item in tqdm(ds, desc="encode ROI"):
        roi = crop_bbox(Image.open(item["image_path"]), item["bbox"])
        vec = enc.encode_pil([roi])[0]          # 512-d
        feats.append(vec); meta.append(item)
    feats=np.stack(feats).astype("float32")
    idx=AnnoyIndex(512,"angular")
    for i,v in enumerate(feats): idx.add_item(i,v)
    idx.build(n_trees)
    idx.save(str(out/"annoy.index"))
    pickle.dump(meta, open(out/"meta.pkl","wb"))
    print("saved index:", out)
