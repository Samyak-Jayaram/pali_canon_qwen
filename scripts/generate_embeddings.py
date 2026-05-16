import json
import numpy as np
import pickle
import time
import re
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from rank_bm25 import BM25Okapi


# ── Config ──────────────────────────────────────────────────────────────────
MODEL_NAME   = "all-mpnet-base-v2"
BATCH_SIZE   = 64

DATA_FILE    = Path("data/chunks.jsonl")
EMBED_DIR    = Path("embeddings")

INDEX_FILE   = EMBED_DIR / "faiss.index"
META_FILE    = EMBED_DIR / "metadata.pkl"
BM25_FILE    = EMBED_DIR / "bm25.pkl"

# Hybrid weights (tune later)
DENSE_WEIGHT  = 0.6
SPARSE_WEIGHT = 0.4


# ── Utils ───────────────────────────────────────────────────────────────────

def simple_tokenize(text: str):
    """Lowercase + basic tokenization (fast, sufficient for BM25)."""
    return re.findall(r"\w+", text.lower())


# ── Load chunks ─────────────────────────────────────────────────────────────

def load_chunks(path: Path):
    texts, metas = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            texts.append(chunk["text"])
            metas.append(chunk)
    return texts, metas


# ── Embedding ───────────────────────────────────────────────────────────────

def embed_in_batches(model, texts):
    all_embeddings = []
    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in tqdm(range(0, len(texts), BATCH_SIZE),
                  total=total_batches, desc="Embedding"):
        batch = texts[i : i + BATCH_SIZE]
        emb = model.encode(
            batch,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        all_embeddings.append(emb)

    return np.vstack(all_embeddings).astype("float32")


# ── FAISS index ─────────────────────────────────────────────────────────────

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index = faiss.IndexIDMap(index)

    ids = np.arange(len(embeddings), dtype=np.int64)
    index.add_with_ids(embeddings, ids)

    return index


# ── BM25 index ──────────────────────────────────────────────────────────────

def build_bm25(texts):
    print("\nBuilding BM25 index...")
    tokenized = [simple_tokenize(t) for t in tqdm(texts, desc="Tokenizing")]
    bm25 = BM25Okapi(tokenized)
    return bm25, tokenized


# ── Hybrid search (for testing / reuse later) ────────────────────────────────

def hybrid_search(query, model, index, bm25, tokenized_corpus, top_k=5):
    """
    Combines:
    - Dense similarity (FAISS)
    - Sparse BM25 score

    Returns ranked indices + scores
    """

    # Dense search
    q_emb = model.encode([query], normalize_embeddings=True)
    D, I = index.search(q_emb.astype("float32"), top_k * 5)

    dense_scores = D[0]
    dense_ids = I[0]

    # Sparse search
    tokenized_query = simple_tokenize(query)
    sparse_scores = bm25.get_scores(tokenized_query)

    # Normalize scores
    dense_scores = (dense_scores - dense_scores.min()) / (dense_scores.ptp() + 1e-6)
    sparse_scores = (sparse_scores - sparse_scores.min()) / (sparse_scores.ptp() + 1e-6)

    # Combine
    hybrid_scores = []

    for idx, d_score in zip(dense_ids, dense_scores):
        s_score = sparse_scores[idx]
        score = DENSE_WEIGHT * d_score + SPARSE_WEIGHT * s_score
        hybrid_scores.append((idx, score))

    # Sort final
    hybrid_scores.sort(key=lambda x: x[1], reverse=True)

    return hybrid_scores[:top_k]


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    EMBED_DIR.mkdir(exist_ok=True)

    # 1. Load
    print("Loading chunks...")
    texts, metas = load_chunks(DATA_FILE)
    print(f"   {len(texts):,} chunks loaded")

    # 2. Model
    print(f"\nLoading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    print(f"   Embedding dim: {model.get_sentence_embedding_dimension()}")

    # 3. Dense embeddings
    print(f"\nEmbedding {len(texts):,} chunks...")
    t0 = time.time()
    embeddings = embed_in_batches(model, texts)
    print(f"   Done in {time.time() - t0:.1f}s")

    # 4. FAISS
    print("\nBuilding FAISS index...")
    index = build_faiss_index(embeddings)
    faiss.write_index(index, str(INDEX_FILE))
    print(f"   Saved → {INDEX_FILE}")

    # 5. BM25
    bm25, tokenized_corpus = build_bm25(texts)
    with open(BM25_FILE, "wb") as f:
        pickle.dump((bm25, tokenized_corpus), f)
    print(f"   Saved → {BM25_FILE}")

    # 6. Metadata
    with open(META_FILE, "wb") as f:
        pickle.dump(metas, f)
    print(f"   Saved → {META_FILE}")

    print(f"\nHybrid index ready")
    print(f"   Dense vectors: {index.ntotal:,}")
    print(f"\nNext step: integrate hybrid_search into your query pipeline")


if __name__ == "__main__":
    main()