import pandas as pd
import numpy as np
from pathlib import Path

CHUNKS_PATH = Path("rag/chunks.csv")
OUT_PATH = Path("data/rag_eval_questions_template.csv")

N = 120
SEED = 42

def main():
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError("rag/chunks.csv not found. Run: python rag/index_pipeline_sklearn.py")

    df = pd.read_csv(CHUNKS_PATH)
    rng = np.random.default_rng(SEED)

    # sample diverse chunks by product if possible
    if "product" in df.columns:
        samples = []
        for prod, g in df.groupby("product"):
            if len(samples) >= N:
                break
            take = min(max(1, N // max(1, df["product"].nunique())), len(g))
            idx = rng.choice(g.index.to_numpy(), size=take, replace=False)
            samples.append(df.loc[idx])
        sampled = pd.concat(samples).sample(n=min(N, sum(len(s) for s in samples)), random_state=SEED)
    else:
        sampled = df.sample(n=min(N, len(df)), random_state=SEED)

    # Build template: you will write "question" manually based on the chunk text
    template = pd.DataFrame({
        "qid": range(1, len(sampled) + 1),
        "question": [""] * len(sampled),
        "expected_chunk_id": sampled["chunk_id"].astype(int).tolist(),
        "expected_row_id": sampled["row_id"].astype(int).tolist() if "row_id" in sampled.columns else [""] * len(sampled),
        "notes": sampled["text"].str.slice(0, 220).tolist()  # paste clue text to help you write the question
    })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    template.to_csv(OUT_PATH, index=False)
    print("Wrote:", OUT_PATH)
    print("Next: open it, fill 'question', then save as data/rag_eval_questions.csv")

if __name__ == "__main__":
    main()
