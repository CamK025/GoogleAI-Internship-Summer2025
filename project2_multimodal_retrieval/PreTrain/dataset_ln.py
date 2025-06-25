from __future__ import annotations
import json, re
from pathlib import Path
from typing import Iterator, Dict, List
from PIL import Image
from utils_bbox import traces_to_bbox
from nltk.corpus import stopwords
STOP = set(stopwords.words("english"))
TOKEN = re.compile(r"[A-Za-z]+")

def valid_utt(txt:str):
    toks = TOKEN.findall(txt.lower())
    return len(toks) >= 2 and any(t not in STOP for t in toks)

class LNIterableDataset:
    """yield {image_path, caption, bbox}  One valid utterance each."""
    def __init__(self, jsonl_path, images_dir, max_items=None):
        self.jsonl_path = Path(jsonl_path)
        self.images_dir = Path(images_dir)
        self.max_items  = max_items
    def __iter__(self)->Iterator[Dict]:
        n = 0
        with self.jsonl_path.open(encoding="utf-8") as fh:
            for line in fh:
                if self.max_items and n >= self.max_items:
                    return
                try:
                    rec = json.loads(line)
                except:
                    continue
                caption = rec["caption"]
                img_id  = rec["image_id"]
                img_path=None
                for ext in(".jpg",".jpeg",".png"):
                    p=self.images_dir/f"{img_id}{ext}"
                    if p.exists(): img_path=p; break
                if not img_path: continue
                w,h=Image.open(img_path).size

                for seg in rec.get("timed_caption",[]):
                    utt = seg.get("utterance","").strip()
                    if not valid_utt(utt): continue
                    start = seg.get("start", seg.get("start_time"))
                    end   = seg.get("end",   seg.get("end_time"))
                    if start is None or end is None: continue

                    raw_tr = rec.get("traces", [])
                    flat = [pt for sub in raw_tr for pt in (sub if isinstance(sub,list) else [sub])]
                    pts = [pt for pt in flat if start<=pt["t"]<=end]
                    if not pts: continue
                    bbox = traces_to_bbox(pts,w,h)
                    yield {"image_path":str(img_path),
                           "caption":caption,
                           "bbox":bbox}
                    n+=1
                    if self.max_items and n>=self.max_items:
                        return
