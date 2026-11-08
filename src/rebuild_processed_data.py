"""Rebuild the two processed data files from the raw UCI download (for a fresh clone).

The raw UCI archive is not part of the repository (docs/DATA_SOURCE.md has the download link and SHA-256).
After downloading and extracting it, run:

    python src/rebuild_processed_data.py

It writes data/processed/cleaned_all_rows.csv with the same recoding code as src/s02_clean_and_audit.py, and
data/processed/train_rows.csv from the committed data/splits/split_assignment.csv, then checks every SHA-256
against the values recorded when the split was frozen. It never overwrites an existing file. Use --out-dir to
write somewhere else, for example to test the rebuild without touching the frozen files.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd  # noqa: E402

from prep import (ALL_COLUMNS, PROCESSED_DIR, RAW_CSV, SPLITS_DIR, add_identifiers, read_raw_as_text,  # noqa: E402
                  recode, sha256_file)

EXPECTED = {
    "raw CSV": "7889d9d0beb7dd1ccc58da99f72763f16afb259b5dbbaa086f8195366ff66137",
    "split_assignment.csv": "8f0077e57e4b59b868929413446b3b445a3be9807b4bb1f4645dad221817b31e",
    "cleaned_all_rows.csv": "2678581eedfbffae3d1d2a4668b3cbfffca78bd5369bcbc2a6981ac4cd810ff8",
    "train_rows.csv": "05dfd30e3ff96f102d95e4f5198037071f5e4484b87f695f8635a55a6e1370e8",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-dir", type=Path, default=PROCESSED_DIR)
    out = parser.parse_args().out_dir
    targets = {"cleaned_all_rows.csv": out / "cleaned_all_rows.csv", "train_rows.csv": out / "train_rows.csv"}
    existing = [name for name, path in targets.items() if path.exists()]
    if existing:
        sys.exit(f"Nothing done: {existing} already exist in {out} (this script never overwrites).")
    assign_path = SPLITS_DIR / "split_assignment.csv"
    for label, path in (("raw CSV", RAW_CSV), ("split_assignment.csv", assign_path)):
        if not path.exists():
            sys.exit(f"Missing input: {path}. See docs/DATA_SOURCE.md for the raw download.")
        if sha256_file(path) != EXPECTED[label]:
            sys.exit(f"{label} does not match its recorded SHA-256; refusing to continue.")
    df = add_identifiers(recode(read_raw_as_text()))          # same code path as src/s02_clean_and_audit.py
    assign = pd.read_csv(assign_path)
    assert (assign["row_id"].to_numpy() == df["row_id"].to_numpy()).all(), "row order differs from the split assignment"
    assert (assign["profile_id"].to_numpy() == df["profile_id"].to_numpy()).all(), "profile ids differ from the split assignment"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(targets["cleaned_all_rows.csv"], index=False)
    train = df[assign["partition"].to_numpy() == "train"].reset_index(drop=True)
    train[["row_id", "record_id", "profile_id", "is_first_copy"] + ALL_COLUMNS].to_csv(targets["train_rows.csv"], index=False)
    ok = True
    for name, path in targets.items():
        same = sha256_file(path) == EXPECTED[name]
        ok &= same
        print(f"{name}: {len(pd.read_csv(path))} rows, {'matches' if same else 'DIFFERS from'} the recorded SHA-256")
    if not ok:
        sys.exit("A rebuilt file differs from its recorded SHA-256.")


if __name__ == "__main__":
    main()
