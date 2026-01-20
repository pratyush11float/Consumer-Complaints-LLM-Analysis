from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
import chromadb


TEXT_COL = "narrative"
LABEL_COL = "Product"
DATE_COL  = "Date received"


def clean_minimal(s: str) -> str:
    if s is None:
        return ""
    return " ".join(str(s).split())


def chunk_text(text: str, chunk_tokens: int = 400, overlap: int = 60) -> list[str]:
    """
    Simple deterministic whitespace chunker.
    'tokens' ~ words (good enough for baseline RAG).
    """
    tokens = text.split()
    if not tokens:
        return []
    chunks = []
    step = max(1, chunk_tokens - overlap)
    i = 0
    while i < len(tokens):
        chunks.append(" ".join(tokens[i:i + chunk_tokens]))
        i += step
    return chunks


@dataclass
class IndexConfig:
    raw_dir: str = "data/raw"
    out_dir: str = "rag"
    collection_name: str = "consumer_complaints"
    model_name: str = "all-MiniLM-L6-v2"
    max_rows: int = 200000
    chunk_tokens: int = 400
    overlap: int = 60
    batch_size: int = 64


def load_biggest_csv(raw_dir: str) -> pd.DataFrame:
    raw = Path(raw_dir)
    csvs = sorted(raw.glob("*.csv"), key=lambda p: p.stat().st_size, reverse=True)
    if not csvs:
        raise FileNotFoundError(f"No CSV found in {raw_dir}. Run scripts/download_data.py")
    return pd.read_csv(csvs[0], low_memory=False)


def batched(iterable, n: int):
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]


def main(cfg: IndexConfig = IndexConfig()):
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load
    df = load_b_
