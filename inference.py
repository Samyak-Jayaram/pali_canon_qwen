import torch
import re
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig

# ── config ────────────────────────────────────────────────────────────────────
MODEL_PATH      = r"trained_models\qwen-0.5-v1"

FAISS_INDEX     = r"embeddings_all-mpnet-base-v2_bm25/faiss.index"
METADATA_PATH   = r"embeddings_all-mpnet-base-v2_bm25/metadata.pkl"
BM25_PATH       = r"embeddings_all-mpnet-base-v2_bm25\bm25.pkl"

TOP_K           = 3
MAX_NEW_TOKENS  = 256
CHUNK_CHAR_CAP  = 800

# Hybrid retrieval tuning
DENSE_WEIGHT    = 0.6
SPARSE_WEIGHT   = 0.4
CANDIDATE_MULT  = 5
SCORE_THRESHOLD = 0.60 

# MUST match training
SYSTEM_PROMPT = (
    "You are a knowledgeable Buddhist scholar specializing in the Pali Canon. "
    "Answer strictly from the provided sutta context. "
    "Only cite sutta UIDs that appear in the context passages."
)

STOP_STRINGS = [
    "\n### User:",
    "\n### Context:",
    "\nHuman:",
    "\nUser:",
    "\n\n\n",
    "Human:",
    "Human resources",
]

# ── load retriever ────────────────────────────────────────────────────────────
print("Loading retriever...")


embedder = SentenceTransformer("all-mpnet-base-v2")

index = faiss.read_index(FAISS_INDEX)

with open(METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)

# BM25
print("Loading BM25...")
try:
    with open(BM25_PATH, "rb") as f:
        bm25, tokenized_corpus = pickle.load(f)
    HAS_BM25 = True
    print("BM25 loaded.")
except Exception:
    HAS_BM25 = False
    print("BM25 not found — using dense-only retrieval.")

sample = metadata[0]
assert "text" in sample, "metadata missing 'text' field"
print(f"Metadata OK — {len(metadata):,} chunks")

# ── load LLM ──────────────────────────────────────────────────────────────────
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto"
)
model.eval()

print("Ready.\n")

# ── utils ─────────────────────────────────────────────────────────────────────
def simple_tokenize(text: str):
    return re.findall(r"\w+", text.lower())


# ── hybrid retrieval ──────────────────────────────────────────────────────────
def retrieve(question: str, top_k: int = TOP_K):
    # Dense
    query_vec = embedder.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype(np.float32)

    candidate_k = top_k * CANDIDATE_MULT
    distances, indices = index.search(query_vec, candidate_k)

    dense_scores = distances[0]
    dense_ids    = indices[0]

    # Normalize dense
    if len(dense_scores) > 0:
        d_min, d_max = dense_scores.min(), dense_scores.max()
        dense_scores = (dense_scores - d_min) / (d_max - d_min + 1e-6)

    # Sparse
    if HAS_BM25:
        tokenized_query = simple_tokenize(question)
        sparse_scores = bm25.get_scores(tokenized_query)

        s_min, s_max = sparse_scores.min(), sparse_scores.max()
        sparse_scores = (sparse_scores - s_min) / (s_max - s_min + 1e-6)
    else:
        sparse_scores = None

    # Fusion
    results = []
    for idx, d_score in zip(dense_ids, dense_scores):
        if idx == -1:
            continue

        if HAS_BM25:
            s_score = sparse_scores[idx]
            score = DENSE_WEIGHT * d_score + SPARSE_WEIGHT * s_score
        else:
            score = float(d_score)

        if score < SCORE_THRESHOLD:
            continue

        chunk = metadata[idx].copy()
        chunk["_score"] = float(score)
        results.append(chunk)

    results.sort(key=lambda x: x["_score"], reverse=True)
    return results[:top_k]


# ── prompt builder ────────────────────────────────────────────────────────────
def build_prompt(question: str, chunks):
    context_parts = []

    for chunk in chunks:
        text = (chunk.get("text") or "")[:CHUNK_CHAR_CAP]
        if not text:
            continue

        uid   = chunk.get("uid", "")
        title = chunk.get("title", "")
        label = f"[{uid} — {title}]\n" if uid else ""

        context_parts.append(f"{label}{text}")

    context_block = "\n\n".join(context_parts)

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"### User:\n{question}\n\n"
        f"### Context:\n{context_block}\n\n"
        f"### Assistant:\n"
    )


# ── post-processing ───────────────────────────────────────────────────────────
def truncate_at_stop_strings(text: str):
    for stop in STOP_STRINGS:
        if stop in text:
            text = text.split(stop)[0]
    return text.strip()


def extract_final_answer(text: str):
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL).strip()

def remove_citation_loops(text: str):
    seen = set()

    def _replace(m):
        key = m.group(0).lower()
        if key in seen:
            return ""
        seen.add(key)
        return m.group(0)

    return re.sub(
        r"\[(?:[Mm][Nn]|[Ss][Nn]|[Aa][Nn]|[Dd][Nn]|[Uu][Dd]|[Ii]ti|[Ss]n[Pp]?|[Tt]hag|[Tt]hig)\s*[\d.]+\]",
        _replace,
        text
    ).strip()


def truncate_to_last_sentence(text: str, max_chars=800):
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]

    last_end = max(
        truncated.rfind(". "),
        truncated.rfind("? "),
        truncated.rfind("! "),
    )

    if last_end > max_chars // 2:
        return truncated[:last_end + 1].strip()

    return truncated.strip()


def post_process(raw: str):
    if "### Assistant:" in raw:
        raw = raw.split("### Assistant:")[-1]

    raw = truncate_at_stop_strings(raw)
    raw = extract_final_answer(raw)
    raw = remove_citation_loops(raw)
    raw = truncate_to_last_sentence(raw)

    return raw


# ── generation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def ask(question: str, show_thinking=False, top_k=TOP_K):
    chunks = retrieve(question, top_k)

    if not chunks:
        return {
            "question": question,
            "answer": "Not found in context — no sufficiently relevant passages retrieved.",
            "sources": [],
        }

    prompt = build_prompt(question, chunks)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        repetition_penalty=1.15,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    raw = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {
        "question": question,
        "answer": post_process(raw),
        "sources": _format_sources(chunks),
    }


def _format_sources(chunks):
    return [
        {
            "uid": c.get("uid", "?"),
            "title": c.get("title", ""),
            "score": c["_score"],
        }
        for c in chunks
    ]

# ── example ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    questions = [
        "What is the meaning of life?",
    ]

    for q in questions:
        result = ask(q)

        print(f"Q: {result['question']}\n")

        print("Sources retrieved:")
        for s in result["sources"]:
            print(f"  [{s['uid']}] {s['title']}  (score: {s['score']:.3f})")

        if not result["sources"]:
            print("  (none above threshold)")


        print(f"\nAnswer:\n{result['answer']}")
        print("\n" + "─" * 60 + "\n")




