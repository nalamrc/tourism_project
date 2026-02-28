
import os
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi
from sklearn.model_selection import train_test_split

HF_USERNAME = os.getenv("HF_USERNAME", "nalamrc")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", f"{HF_USERNAME}/tourism-wellness-dataset")
HF_TOKEN = os.getenv("HF_TOKEN", "")
TARGET_COL = "ProdTaken"
BASE_DIR = Path(os.getenv("GITHUB_SRC_ART_BASE_DIR", Path(__file__).resolve().parents[1])).expanduser()
RAW_DATA_PATH = os.getenv("RAW_DATA_PATH", "")

api = HfApi(token=HF_TOKEN) if HF_TOKEN else None

data_dir = BASE_DIR / "data"
data_dir.mkdir(parents=True, exist_ok=True)

if HF_TOKEN and "nalamrc" not in HF_DATASET_REPO:
    raw_df = load_dataset(HF_DATASET_REPO, data_files="tourism.csv", split="train").to_pandas()
else:
    # Prefer explicit path from CI/local env, then common project locations.
    raw_candidates = []
    if RAW_DATA_PATH:
        raw_candidates.append(Path(RAW_DATA_PATH))
    raw_candidates.extend([
        BASE_DIR / "data" / "tourism.csv",
        BASE_DIR / "tourism.csv",
        Path("data/tourism.csv"),
        Path("tourism.csv"),
    ])
    raw_path = next((p for p in raw_candidates if p.exists()), None)
    if raw_path is None:
        searched = "
".join(str(p) for p in raw_candidates)
        raise FileNotFoundError(f"Raw dataset not found. Searched:
{searched}")
    raw_df = pd.read_csv(raw_path)

# Remove unnamed index-like column and customer id from modeling inputs.
unnamed_cols = [c for c in raw_df.columns if str(c).startswith("Unnamed") or str(c).strip() == ""]
clean_df = raw_df.drop(columns=unnamed_cols, errors="ignore").drop(columns=["CustomerID"], errors="ignore")

# Standardize object values for consistent categorical encoding.
for col in clean_df.select_dtypes(include="object").columns:
    clean_df[col] = clean_df[col].astype(str).str.strip()

X = clean_df.drop(columns=[TARGET_COL])
y = clean_df[TARGET_COL].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

train_df = X_train.copy()
train_df[TARGET_COL] = y_train.values

test_df = X_test.copy()
test_df[TARGET_COL] = y_test.values

train_path = data_dir / "train.csv"
test_path = data_dir / "test.csv"
train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

if api and HF_TOKEN and "nalamrc" not in HF_DATASET_REPO:
    api.upload_file(path_or_fileobj=str(train_path), path_in_repo="train.csv", repo_id=HF_DATASET_REPO, repo_type="dataset")
    api.upload_file(path_or_fileobj=str(test_path), path_in_repo="test.csv", repo_id=HF_DATASET_REPO, repo_type="dataset")

pd.DataFrame([
    {"artifact": "train.csv", "rows": len(train_df), "positive_rate": float(train_df[TARGET_COL].mean())},
    {"artifact": "test.csv", "rows": len(test_df), "positive_rate": float(test_df[TARGET_COL].mean())},
])
