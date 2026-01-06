from pathlib import Path
import shutil
import kagglehub

DATASET = "namigabbasov/consumer-complaint-dataset"

def main():
    # Downloads to kagglehub cache and returns that directory path
    cache_dir = Path(kagglehub.dataset_download(DATASET))
    print("KaggleHub cache dir:", cache_dir)

    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)

    csvs = list(cache_dir.rglob("*.csv"))
    if not csvs:
        raise RuntimeError(f"No CSV found under {cache_dir}. Found: {list(cache_dir.rglob('*'))[:20]}")

    for f in csvs:
        dest = out_dir / f.name
        shutil.copyfile(f, dest)
        print("Copied:", f.name, "->", dest)

    print("\nDone. Raw data saved to data/raw/ (gitignored).")

if __name__ == "__main__":
    main()
