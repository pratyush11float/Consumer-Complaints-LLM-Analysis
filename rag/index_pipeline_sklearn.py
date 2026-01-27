from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

TEXT_COL = "narrative"
LABEL_COL = "Product"
DATE_COL  = "Date received"


def clean_minimal(s: str) -> str:
    if s is None:
        return ""
    return " ".join(str(s).split())


def chunk_text(text: str, chunk_words: int = 400, overlap: int = 60) -> list[str]:
    """
    Deterministic whitespace chunker.
    'chunk_words' approximates 300–500 tokens.
    """
    tokens = text.split()
    if not tokens:
        return []
    step = max(1, chunk_words - overlap)
    chunks = []
    i = 0
    while i < len(tokens):
        chunks.append(" ".join(tokens[i:i + chunk_words]))
        i += step
    return chunks


@dataclass
class IndexConfig:
    raw_dir: str = "data/raw"
    out_dir: str = "rag"
    model_name: str = "all-MiniLM-L6-v2"
    max_rows: int = 120000      # keep manageable on laptop
    chunk_words: int = 400
    overlap: int = 60
    batch_size: int = 64


def load_biggest_csv(raw_dir: str) -> pd.DataFrame:
    raw = Path(raw_dir)
    csvs = sorted(raw.glob("*.csv"), key=lambda p: p.stat().st_size, reverse=True)
    if not csvs:
        raise FileNotFoundError(f"No CSV found in {raw_dir}. Run scripts/download_data.py")
    return pd.read_csv(csvs[0], low_memory=False)


def main(cfg: IndexConfig = IndexConfig()):
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load
    df = load_biggest_csv(cfg.raw_dir)
    df = df[[TEXT_COL, LABEL_COL, DATE_COL]].copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[TEXT_COL, DATE_COL]).copy()

    # Keep manageable
    if len(df) > cfg.max_rows:
        df = df.sample(n=cfg.max_rows, random_state=42).copy()

    df[TEXT_COL] = df[TEXT_COL].map(clean_minimal)

    # Chunk
    records = []
    chunk_id = 0
    for row_id, row in df.reset_index(drop=True).iterrows():
        for ch in chunk_text(row[TEXT_COL], cfg.chunk_words, cfg.overlap):
            records.append({
                "chunk_id": chunk_id,
                "row_id": int(row_id),
                "date_received": row[DATE_COL].date().isoformat(),
                "product": str(row.get(LABEL_COL, "")),
                "text": ch
            })
            chunk_id += 1

    chunks_df = pd.DataFrame(records)
    print("Rows used:", len(df))
    print("Chunks created:", len(chunks_df))

    # Save chunks metadata as CSV (no pyarrow)
    chunks_path = out / "chunks.csv"
    chunks_df.to_csv(chunks_path, index=False)
    print("Saved:", chunks_path)

    # Embed and save embeddings
    model = SentenceTransformer(cfg.model_name)
    embs = model.encode(
        chunks_df["text"].tolist(),
        batch_size=cfg.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True
    ).astype("float32")

    emb_path = out / "embeddings.npy"
    np.save(emb_path, embs)
    print("Saved:", emb_path)

    meta = {
        "backend": "sklearn (NearestNeighbors at query time)",
        "model_name": cfg.model_name,
        "max_rows": cfg.max_rows,
        "chunk_words": cfg.chunk_words,
        "overlap": cfg.overlap,
        "n_chunks": int(len(chunks_df)),
        "embedding_dim": int(embs.shape[1])
    }
    meta_path = out / "index_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print("Saved:", meta_path)

    print("\nNext: run rag/query_smoke_test_sklearn.py to sanity-check retrieval.")


if __name__ == "__main__":
    main()
