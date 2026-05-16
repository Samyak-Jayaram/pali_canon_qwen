import os
import re
import json
import time
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

QUESTIONS_FILE = Path("data/questions.jsonl")
INDEX_FILE     = Path("embeddings_all-MiniLM-L6-v2/faiss.index")
META_FILE      = Path("embeddings_all-MiniLM-L6-v2/metadata.pkl")
CHUNKS_FILE    = Path("data/chunks.jsonl")
COT_OUT        = Path("data/cot_dataset_fixed.jsonl")
DATA_DIR       = Path("data")

MODEL_NAME     = "all-MiniLM-L6-v2"
TOP_K          = 3
API_DELAY      = 0.3

# ── CoT Prompt ────────────────────────────────────────────────────────────────

COT_SYSTEM = """You are a knowledgeable Buddhist scholar specializing in the Pali Canon \
and Early Buddhism. You have deep familiarity with the Nikayas — DN, MN, SN, AN, and KN.

Your role is to answer questions about Buddhist teachings grounded STRICTLY in the \
provided sutta passages.

CITATION RULES — follow these exactly:
- You may ONLY cite the sutta UIDs listed under "Available sources" in the user message.
- Do NOT invent, guess, or recall citations from memory.
- If a sutta is not in the available sources list, do not cite it.
- Format citations as the UID exactly as given (e.g. "an9.35", "mn36").

Always respond in this exact format:
<thinking>
[Reason step by step through the passages. Quote or paraphrase specific text from \
the provided passages to support your reasoning. Identify relevant teachings, connect \
ideas, note important Pali terms. Be thorough — 3 to 6 sentences.]
</thinking>
<answer>
[Clear, concise answer grounded in the passages. Cite only the UIDs from the available \
sources list. If the passages do not contain enough information to answer, say so.]
</answer>"""

COT_USER = """Available sources (you may ONLY cite these UIDs): {uid_list}

Retrieved sutta passages:
{passages}

Question: {question}"""

# ── Citation audit ────────────────────────────────────────────────────────────

CITATION_RE = re.compile(
    r'\b((?:mn|sn|an|dn|kn|ud|iti|snp|thag|thig|mil)\s*\d+(?:\.\d+)?)',
    re.IGNORECASE
)

def extract_cited_uids(text: str) -> set[str]:
    """Extract all sutta citations from a text block, normalised to lowercase no-space."""
    found = CITATION_RE.findall(text)
    return {re.sub(r'\s+', '', uid.lower()) for uid in found}


def normalise_uid(uid: str) -> str:
    return re.sub(r'\s+', '', uid.lower().strip())


def has_hallucinated_citation(answer: str, thinking: str, retrieved_uids: list[str]) -> bool:
    """
    Returns True if the answer or thinking cites a UID not in retrieved_uids.
    """
    allowed = {normalise_uid(u) for u in retrieved_uids if u}
    cited   = extract_cited_uids(answer) | extract_cited_uids(thinking)

    for c in cited:
        # allow partial match — "an9" matches "an9.35"
        if not any(c in a or a in c for a in allowed):
            return True
    return False


# ── API calls ─────────────────────────────────────────────────────────────────

def call_deepseek(question: str, passages: str, uid_list: str) -> str:
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com"
    )
    response = client.chat.completions.create(
        model="deepseek-reasoner",   # R1 — proper grounded reasoning, not V3
        messages=[
            {"role": "system", "content": COT_SYSTEM},
            {"role": "user",   "content": COT_USER.format(
                passages=passages, question=question, uid_list=uid_list
            )}
        ],
        max_tokens=1500,
        # temperature not supported by deepseek-reasoner — omit
    )
    raw = response.choices[0].message.content
    if not raw:
        raise ValueError("DeepSeek returned empty response")
    return raw.strip()


def generate_cot(question: str, passages: str, uid_list: str) -> str:
    return call_deepseek(question, passages, uid_list)


# ── Response parser ───────────────────────────────────────────────────────────

def parse_cot_response(raw: str) -> tuple[str, str]:
    """
    Extract (thinking, answer) from LLM output.
    Handles: R1 native <think> blocks, missing closing tags,
    markdown headers, plain text fallback.
    """
    text = raw.strip()

    # Strip R1's native <think>...</think> block — it sits BEFORE our formatted output
    if "<think>" in text:
        if "</think>" in text:
            text = text.split("</think>", 1)[-1].strip()
        else:
            text = text.split("<think>", 1)[0].strip()

    if "<thinking>" in text and "<answer>" in text:
        thinking = ""
        answer   = ""

        if "</thinking>" in text:
            thinking = text.split("<thinking>", 1)[1].split("</thinking>", 1)[0].strip()
        else:
            thinking = text.split("<thinking>", 1)[1].split("<answer>", 1)[0].strip()

        if "</answer>" in text:
            answer = text.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
        else:
            answer = text.split("<answer>", 1)[1].strip()

        if answer:
            return thinking, answer

    # Only <answer> present
    if "<answer>" in text:
        if "</answer>" in text:
            answer = text.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
        else:
            answer = text.split("<answer>", 1)[1].strip()
        thinking = text.split("<answer>", 1)[0].strip()
        if answer:
            return thinking, answer

    # Markdown bold headers
    md_answer = re.search(r"\*\*Answer[:\*]+\*?\*?\s*\n+(.*?)$", text, re.DOTALL | re.IGNORECASE)
    md_think  = re.search(r"\*\*Thinking[:\*]+\*?\*?\s*\n+(.*?)(?=\*\*Answer|\Z)", text, re.DOTALL | re.IGNORECASE)
    if md_answer:
        return (md_think.group(1).strip() if md_think else ""), md_answer.group(1).strip()

    # Plain text headers
    plain_answer = re.search(r"(?:^|\n)Answer:\s*\n?(.*?)$", text, re.DOTALL | re.IGNORECASE)
    plain_think  = re.search(r"(?:^|\n)Thinking:\s*\n?(.*?)(?=Answer:|\Z)", text, re.DOTALL | re.IGNORECASE)
    if plain_answer:
        return (plain_think.group(1).strip() if plain_think else ""), plain_answer.group(1).strip()

    # Last resort: save whole response as answer
    if len(text) > 30:
        return "", text

    return "", ""


# ── Quality filter ────────────────────────────────────────────────────────────

def is_quality_ok(thinking: str, answer: str, retrieved: list[dict]) -> bool:
    """
    Stricter than original — checks:
    - answer has real content (>20 words)
    - thinking has real reasoning (>30 words)
    - at least one retrieved chunk has actual text
    - thinking references at least one retrieved UID (grounding check)
    """
    if len(answer.split()) < 20:
        return False
    if len(thinking.split()) < 30:
        return False

    texts = [r.get("text", "") for r in retrieved]
    if not any(len(t) > 50 for t in texts):
        return False

    uids = [r.get("uid", "") for r in retrieved]
    if not any(
        normalise_uid(uid) in thinking.lower()
        for uid in uids if uid
    ):
        return False

    return True


# ── Chunk text loader ─────────────────────────────────────────────────────────

def load_chunk_texts(path: Path) -> dict[str, str]:
    lookup = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            cid  = chunk.get("chunk_id")
            text = chunk.get("text", "")
            if cid and text:
                lookup[cid] = text
    print(f"   Loaded {len(lookup):,} chunk texts from {path}")
    return lookup


# ── Retriever ─────────────────────────────────────────────────────────────────

class PaliRetriever:
    def __init__(self, chunk_texts: dict[str, str]):
        print(f"Loading embedding model: {MODEL_NAME}")
        self.model = SentenceTransformer(MODEL_NAME)

        print(f"Loading FAISS index: {INDEX_FILE}")
        self.index = faiss.read_index(str(INDEX_FILE))

        print(f"Loading metadata: {META_FILE}")
        with open(META_FILE, "rb") as f:
            self.metas = pickle.load(f)

        self.chunk_texts = chunk_texts
        print(f"   Index size: {self.index.ntotal:,} vectors")

        sample    = self.metas[0] if self.metas else {}
        sample_id = sample.get("chunk_id", "")
        if not self.chunk_texts.get(sample_id):
            print(f"WARNING: chunk_id '{sample_id}' not found in text lookup!")
        else:
            print(f"   Text lookup OK (sample: '{self.chunk_texts[sample_id][:60]}...')")

    def retrieve(self, question: str, k: int = TOP_K) -> list[dict]:
        vec = self.model.encode([question], normalize_embeddings=True)
        vec = np.array(vec, dtype="float32")
        scores, indices = self.index.search(vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metas):
                continue
            meta = self.metas[idx].copy()
            meta["score"] = float(score)
            cid = meta.get("chunk_id", "")
            meta["text"] = self.chunk_texts.get(cid, "")
            results.append(meta)
        return results

    def format_passages(self, retrieved: list[dict]) -> str:
        parts = []
        for i, r in enumerate(retrieved, 1):
            uid   = r.get("uid", "unknown")
            title = r.get("title", "Unknown Sutta")
            text  = r.get("text", "")
            if not text:
                continue
            parts.append(f"[{i}] {uid} — {title}\n{text}")
        return "\n\n".join(parts) if parts else "[No relevant passages found]"


# ── Resume support ────────────────────────────────────────────────────────────

def load_done_keys(path: Path) -> set[str]:
    done = set()
    if not path.exists():
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line).get("key", ""))
            except Exception:
                pass
    return done


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    DATA_DIR.mkdir(exist_ok=True)

    provider = "DeepSeek-R1 (reasoner)"
    print(f"\nProvider: {provider}")

    print(f"\nLoading questions from {QUESTIONS_FILE}...")
    question_records = []
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                question_records.append(json.loads(line))

    all_pairs = []
    for rec in question_records:
        for q_idx, question in enumerate(rec["questions"]):
            all_pairs.append({
                "key":          f"{rec['chunk_id']}::{q_idx}",
                "chunk_id":     rec["chunk_id"],
                "uid":          rec["uid"],
                "title":        rec["title"],
                "nikaya":       rec.get("nikaya", ""),
                "source_chunk": rec["chunk_text"],
                "question":     question,
            })

    print(f"   Total training pairs: {len(all_pairs):,}")

    done_keys = load_done_keys(COT_OUT)
    todo = [p for p in all_pairs if p["key"] not in done_keys]
    print(f"   Already done: {len(done_keys):,} | Remaining: {len(todo):,}")

    if not todo:
        print("All pairs already processed!")
        return

    # Cost estimate — R1 is ~$2.19/M input, $8.19/M output
    avg_tokens = 2000
    est = (len(todo) * avg_tokens / 1_000_000) * 5.0   # blended R1 rate
    print(f"   Estimated DeepSeek-R1 cost: ~${est:.2f}")

    print(f"\nLoading chunk texts from {CHUNKS_FILE}...")
    chunk_texts = load_chunk_texts(CHUNKS_FILE)
    retriever   = PaliRetriever(chunk_texts)

    errors          = 0
    hallucination_count = 0
    quality_fail_count  = 0

    with open(COT_OUT, "a", encoding="utf-8") as out_f:
        for pair in tqdm(todo, desc="Generating CoT answers"):
            question = pair["question"]
            key      = pair["key"]

            try:
                # Retrieve
                retrieved = retriever.retrieve(question, k=TOP_K)
                passages  = retriever.format_passages(retrieved)

                # Build uid_list string — injected into prompt so model knows
                # exactly which UIDs it is allowed to cite
                retrieved_uids = [r.get("uid", "") for r in retrieved if r.get("uid")]
                uid_list = ", ".join(retrieved_uids) if retrieved_uids else "none"

                # Generate
                raw = generate_cot(question, passages, uid_list)
                thinking, answer = parse_cot_response(raw)

                # Citation audit
                hallucinated = has_hallucinated_citation(answer, thinking, retrieved_uids)
                if hallucinated:
                    hallucination_count += 1
                    # One retry with a stricter reminder
                    tqdm.write(f"  Hallucinated citation on {key} — retrying...")
                    time.sleep(API_DELAY)
                    try:
                        raw2 = generate_cot(question, passages, uid_list)
                        t2, a2 = parse_cot_response(raw2)
                        if not has_hallucinated_citation(a2, t2, retrieved_uids):
                            thinking, answer = t2, a2
                            hallucinated = False
                            tqdm.write(f"  Retry succeeded for {key}")
                        else:
                            tqdm.write(f"  Retry still hallucinated for {key} — saving as quality_ok=False")
                    except Exception as retry_err:
                        tqdm.write(f"  Retry failed for {key}: {retry_err}")

                # Quality gate
                quality_ok = (
                    is_quality_ok(thinking, answer, retrieved)
                    and not hallucinated
                )
                if not quality_ok:
                    quality_fail_count += 1

                retrieved_texts = [
                    {"uid": r.get("uid"), "title": r.get("title"), "text": r.get("text", "")}
                    for r in retrieved
                ]

                completion_text = (
                    f"<thinking>\n{thinking}\n</thinking>\n"
                    f"<answer>\n{answer}\n</answer>"
                )

                record = {
                    "key":              key,
                    "chunk_id":         pair["chunk_id"],
                    "uid":              pair["uid"],
                    "title":            pair["title"],
                    "nikaya":           pair["nikaya"],
                    "question":         question,
                    "retrieved_chunks": retrieved_texts,
                    "prompt":           COT_USER.format(
                                            passages=passages,
                                            question=question,
                                            uid_list=uid_list
                                        ),
                    "completion":       completion_text,
                    "thinking":         thinking,
                    "answer":           answer,
                    "quality_ok":       quality_ok,
                    "hallucinated":     hallucinated,   # explicit flag for auditing
                }

                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                time.sleep(API_DELAY)

            except Exception as e:
                errors += 1
                err_str = str(e).lower()
                if "rate" in err_str or "429" in err_str:
                    tqdm.write(f"Rate limit — waiting 60s... ({key})")
                    time.sleep(60)
                else:
                    tqdm.write(f"Error on {key}: {e}")
                    time.sleep(API_DELAY * 4)

    total = len(done_keys) + len(todo) - errors
    print(f"\nDone!")
    print(f"   Records saved:              {total:,}")
    print(f"   Hallucinated citations:     {hallucination_count:,} ({100*hallucination_count/max(len(todo),1):.1f}%)")
    print(f"   quality_ok=False:           {quality_fail_count:,} ({100*quality_fail_count/max(len(todo),1):.1f}%)")
    print(f"   API errors:                 {errors:,}")
    print(f"\nNext: filter quality_ok=True records in split_dataset.py")
    print(f"      then re-run train.py")


if __name__ == "__main__":
    main()