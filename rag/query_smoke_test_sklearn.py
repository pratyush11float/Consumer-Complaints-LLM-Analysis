import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

CHUNKS_PATH = "rag/chunks.csv"
EMB_PATH = "rag/embeddings.npy"
MODEL_NAME = "all-MiniLM-L6-v2"

chunks = pd.read_csv(CHUNKS_PATH)
embs = np.load(EMB_PATH)

nn = NearestNeighbors(n_neighbors=5, metric="cosine", algorithm="brute")
nn.fit(embs)

embedder = SentenceTransformer(MODEL_NAME)

q = "credit card charged twice and dispute not resolved"
q_emb = embedder.encode([q], normalize_embeddings=True).astype("float32")

dist, idx = nn.kneighbors(q_emb, n_neighbors=5)

for rank, i in enumerate(idx[0], 1):
    print("\n---", rank, "cosine_dist=", float(dist[0][rank-1]))
    row = chunks.iloc[i]
    print("chunk_id:", row["chunk_id"], "| product:", row["product"], "| date:", row["date_received"])
    print(row["text"][:320], "...")
