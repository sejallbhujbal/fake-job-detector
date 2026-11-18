"""
Add new labelled posts to data/custom_india/custom_india.csv.

Usage
-----
Interactive:        python data/custom_india/add_sample.py
Single sample:      python data/custom_india/add_sample.py --text "..." --label 1 [--company ..]
Bulk import:        python data/custom_india/add_sample.py --from-csv collected.csv
Drop synthetic:     python data/custom_india/add_sample.py --drop-synthetic

The script validates the label, strips whitespace, and skips exact duplicate texts.
Only the Python standard library + pandas are needed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "custom_india.csv"
COLUMNS = ["text", "company", "email", "salary", "source", "label"]


def load() -> pd.DataFrame:
    if CSV_PATH.exists():
        return pd.read_csv(CSV_PATH, dtype=str, keep_default_na=False)
    return pd.DataFrame(columns=COLUMNS)


def save(df: pd.DataFrame) -> None:
    df[COLUMNS].to_csv(CSV_PATH, index=False, encoding="utf-8")


def normalise(row: dict) -> dict:
    """Clean a single row and validate it. Raises ValueError if invalid."""
    clean = {c: str(row.get(c, "") or "").strip() for c in COLUMNS}
    if len(clean["text"].split()) < 3:
        raise ValueError("text must contain at least 3 words")
    if clean["label"] not in {"0", "1"}:
        raise ValueError("label must be 0 (real) or 1 (fake)")
    clean["source"] = clean["source"] or "manual"
    return clean


def add_rows(rows: list[dict]) -> tuple[int, int]:
    """Append rows, skipping duplicates. Returns (added, skipped)."""
    df = load()
    existing = set(df["text"].str.strip())
    new, skipped = [], 0
    for r in rows:
        try:
            r = normalise(r)
        except ValueError as exc:
            print(f"  skipped (invalid: {exc}): {str(r.get('text', ''))[:60]!r}")
            skipped += 1
            continue
        if r["text"] in existing:
            print(f"  skipped (duplicate): {r['text'][:60]!r}")
            skipped += 1
            continue
        existing.add(r["text"])
        new.append(r)
    if new:
        df = pd.concat([df, pd.DataFrame(new)], ignore_index=True)
        save(df)
    return len(new), skipped


def interactive() -> dict:
    print("Paste the post text. Finish with an empty line:")
    lines = []
    while True:
        line = input()
        if not line.strip():
            break
        lines.append(line)
    return {
        "text": " ".join(lines),
        "company": input("Company name (optional): "),
        "email": input("Contact e-mail (optional): "),
        "salary": input("Salary / stipend as written (optional): "),
        "source": input("Source (e.g. internshala, linkedin, whatsapp_forward): "),
        "label": input("Label - 1 = FAKE, 0 = REAL: "),
    }


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--text")
    p.add_argument("--company", default="")
    p.add_argument("--email", default="")
    p.add_argument("--salary", default="")
    p.add_argument("--source", default="manual")
    p.add_argument("--label", choices=["0", "1"])
    p.add_argument("--from-csv", type=Path, help="bulk-import a CSV with the same columns")
    p.add_argument("--drop-synthetic", action="store_true", help="remove synthetic_example rows")
    args = p.parse_args(argv)

    if args.drop_synthetic:
        df = load()
        before = len(df)
        df = df[df["source"] != "synthetic_example"]
        save(df)
        print(f"Removed {before - len(df)} synthetic rows; {len(df)} rows remain.")
        return

    if args.from_csv:
        rows = pd.read_csv(args.from_csv, dtype=str, keep_default_na=False).to_dict("records")
    elif args.text:
        if args.label is None:
            sys.exit("--label is required with --text")
        rows = [vars(args)]
    else:
        rows = [interactive()]

    added, skipped = add_rows(rows)
    print(f"Added {added} row(s), skipped {skipped}. File: {CSV_PATH}")


if __name__ == "__main__":
    main()
