"""
Dataset loaders.

1. EMSCAD - "Real or Fake Job Posting Prediction" (Kaggle, ~17,880 posts,
   ~4.8 % fraudulent).
   Download: https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction
   Put `fake_job_postings.csv` in data/raw/  (or run: python -m src.data_loader --download)

2. Custom Indian dataset - data/custom_india/custom_india.csv
   Columns: text, company, email, salary, source, label  (label 1 = fake, 0 = real)

Both are converted to ONE common schema so every model sees the same inputs:
    text, company, email, salary, label
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    CUSTOM_INDIA_CSV, EMSCAD_CSV, KAGGLE_DATASET, RAW_DIR, SEED, TEST_SIZE, VAL_SIZE,
)

EMSCAD_TEXT_COLUMNS = ["title", "company_profile", "description", "requirements", "benefits"]
COMMON_COLUMNS = ["text", "company", "email", "salary", "label"]


def _missing_emscad_message() -> str:
    return (
        f"EMSCAD dataset not found at {EMSCAD_CSV}.\n"
        "Download it from Kaggle:\n"
        "  https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction\n"
        "and place fake_job_postings.csv in data/raw/, or run\n"
        "  python -m src.data_loader --download   (needs `pip install kaggle` + API token)"
    )


def load_emscad_raw(path=EMSCAD_CSV) -> pd.DataFrame:
    """Load the original Kaggle CSV unchanged (used by the EDA notebook)."""
    if not path.exists():
        raise FileNotFoundError(_missing_emscad_message())
    return pd.read_csv(path)


def emscad_to_common(raw: pd.DataFrame) -> pd.DataFrame:
    """Map EMSCAD columns onto the app's 4 input fields.

    * text    = title + company profile + description + requirements + benefits
    * company = first 200 chars of company_profile ('' when missing) - stands in
                for "did the post identify a company?"
    * email   = '' (EMSCAD anonymises e-mails as #EMAIL_...# tokens in text)
    * salary  = salary_range
    NOTE: `has_company_logo` is a strong feature in EMSCAD but a user pasting
    text cannot supply it, so we deliberately do NOT train on it (avoids
    train/serve skew). The EDA notebook still analyses it.
    """
    df = raw.copy()
    for c in EMSCAD_TEXT_COLUMNS + ["salary_range"]:
        df[c] = df[c].fillna("").astype(str)
    out = pd.DataFrame({
        "text": df[EMSCAD_TEXT_COLUMNS].agg(lambda r: "\n".join(x for x in r if x.strip()), axis=1),
        "company": df["company_profile"].str.slice(0, 200),
        "email": "",
        "salary": df["salary_range"],
        "label": df["fraudulent"].astype(int),
    })
    out = out[out["text"].str.strip().str.len() > 0]
    out = out.drop_duplicates(subset="text").reset_index(drop=True)
    return out


def load_emscad(path=EMSCAD_CSV) -> pd.DataFrame:
    """Load EMSCAD in the common schema."""
    return emscad_to_common(load_emscad_raw(path))


def load_custom_india(path=CUSTOM_INDIA_CSV) -> pd.DataFrame:
    """Load the custom Indian dataset in the common schema (+ `source`)."""
    if not path.exists():
        raise FileNotFoundError(f"Custom Indian dataset not found at {path}.")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = set(COMMON_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    df["label"] = df["label"].astype(int)
    if not set(df["label"].unique()) <= {0, 1}:
        raise ValueError("label must be 0 (real) or 1 (fake)")
    return df.reset_index(drop=True)


def stratified_split(df: pd.DataFrame, seed: int = SEED):
    """70 / 15 / 15 train / val / test split that keeps the fake-post ratio
    identical in every part (important with ~5 % positives)."""
    train_val, test = train_test_split(df, test_size=TEST_SIZE, stratify=df["label"], random_state=seed)
    val_rel = VAL_SIZE / (1 - TEST_SIZE)
    train, val = train_test_split(train_val, test_size=val_rel, stratify=train_val["label"], random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def download_emscad() -> None:
    """Download EMSCAD with the Kaggle CLI (requires ~/.kaggle/kaggle.json)."""
    if shutil.which("kaggle") is None:
        sys.exit("Kaggle CLI not found. Run `pip install kaggle` and set up your API token "
                 "(Kaggle > Settings > Create New Token).\n" + _missing_emscad_message())
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(RAW_DIR), "--unzip"],
        check=True,
    )
    print(f"Downloaded to {RAW_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download / inspect datasets")
    parser.add_argument("--download", action="store_true", help="download EMSCAD via Kaggle CLI")
    args = parser.parse_args()
    if args.download:
        download_emscad()
    if EMSCAD_CSV.exists():
        df = load_emscad()
        print(f"EMSCAD: {len(df):,} posts, fake ratio = {df['label'].mean():.3%}")
        tr, va, te = stratified_split(df)
        print(f"split -> train {len(tr):,} | val {len(va):,} | test {len(te):,}")
    else:
        print(_missing_emscad_message())
    if CUSTOM_INDIA_CSV.exists():
        ci = load_custom_india()
        print(f"Custom India: {len(ci)} posts ({ci['label'].sum()} fake, "
              f"{(ci['source'] == 'synthetic_example').sum()} synthetic examples)")


if __name__ == "__main__":
    main()
