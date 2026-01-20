from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer

import faiss  # pip install faiss-cpu


TEXT_COL = "narrative"
LABEL_COL = "Product"
DATE_COL  = "Date received"


def clean_minimal(s: str) -> str:
    if s is None:
        return ""
    return " ".join(str(s).split())


def chunk_text(text: str, chunk_tokens: int = 400, overlap: int = 60) -> list[str]:
    """
    Simple whitespace token chunker (good enough for DS/RAG baseline).
    'tokens' here are approximate (words). Deterministic + fast.
    """
    tokens = text.split()
    if not tokens:
        return []
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i:i+chunk_tokens]
        chunks.append(" ".join(chunk))
        i += max(1, chunk_tokens - overlap)
    return chunks


@dataclass
class IndexConfig:
    dataset_csv: str = "data/raw"      # directory; will pick biggest csv
    out_dir: str = "rag"
    model_name: str = "all-MiniLM-L6-v2"
    max_rows: int = 200000            # keep index size reasonable
    chunk_tokens: int = 400
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

    df = load_biggest_csv(cfg.dataset_csv)
    df = df[[TEXT_COL, LABEL_COL, DATE_COL]].copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[TEXT_COL, DATE_COL]).copy()

    # keep it manageable
    if len(df) > cfg.max_rows:
        df = df.sample(n=cfg.max_rows, random_state=42).copy()

    df[TEXT_COL] = df[TEXT_COL].map(clean_minimal)

    # build chunks + metadata
    records = []
    chunk_id = 0
    for row_id, row in df.reset_index(drop=True).iterrows():
        text = row[TEXT_COL]
        chunks = chunk_text(text, chunk_tokens=cfg.chunk_tokens, overlap=cfg.overlap)

        for ch in chunks:
            records.append({
                "chunk_id": chunk_id,
                "row_id": row_id,                 # complaint row in sampled df
                "date_received": row[DATE_COL],
                "product": row.get(LABEL_COL, None),
                "text": ch
            })
            chunk_id += 1

    chunks_df = pd.DataFrame(records)
    print("Chunks:", len(chunks_df))

    # embed
    model = SentenceTransformer(cfg.model_name)
    embs = model.encode(
        chunks_df["text"].tolist(),
        batch_size=cfg.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True
    ).astype("float32")

    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine when vectors normalized
    index.add(embs)

    # save
    faiss.write_index(index, str(out / "index.faiss"))
    chunks_df.to_parquet(out / "chunks.parquet", index=False)

    meta = {
        "model_name": cfg.model_name,
        "max_rows": cfg.max_rows,
        "chunk_tokens": cfg.chunk_tokens,
        "overlap": cfg.overlap,
        "n_chunks": len(chunks_df),
        "embedding_dim": dim
    }
    pd.Series(meta).to_json(out / "index_metadata.json")

    print("Saved:")
    print(" -", out / "index.faiss")
    print(" -", out / "chunks.parquet")
    print(" -", out / "index_metadata.json")


if __name__ == "__main__":
    main()
