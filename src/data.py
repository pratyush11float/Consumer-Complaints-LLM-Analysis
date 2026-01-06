from pathlib import Path
import pandas as pd

def load_raw_csv(raw_dir: str = "data/raw") -> pd.DataFrame:
    raw_path = Path(raw_dir)
    csvs = list(raw_path.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV in {raw_dir}. Run: python scripts/download_data.py")
    # If multiple CSVs exist, pick the largest
    csvs_sorted = sorted(csvs, key=lambda p: p.stat().st_size, reverse=True)
    df = pd.read_csv(csvs_sorted[0], low_memory=False)
    return df

def pick_first_existing(df: pd.DataFrame, candidates: list[str]) -> str:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    raise KeyError(f"None of these columns found: {candidates}\nAvailable columns: {list(df.columns)}")
