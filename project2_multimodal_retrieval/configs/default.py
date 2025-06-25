from pathlib import Path

# Root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data path
DATASET_DIR  = PROJECT_ROOT / "Dataset"
JSONL_TRAIN  = DATASET_DIR / "filtered_original_train.jsonl"
IMAGES_DIR   = DATASET_DIR / "Original_train"

# Output index
CACHE_DIR    = PROJECT_ROOT / "cache"
INDEX_DIR    = CACHE_DIR / "annoy_idx"
INDEX_FILE   = INDEX_DIR / "annoy.index"
META_FILE    = INDEX_DIR / "meta.pkl"

# hyperparameters
DEVICE_DEFAULT    = "cuda"          
TOPK_DEFAULT      = 5
