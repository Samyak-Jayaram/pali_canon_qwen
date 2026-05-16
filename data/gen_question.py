import os
import re
import json
import time
import random
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

CHUNKS_FILE   = Path("data/chunks.jsonl")    
QUESTIONS_OUT = Path("data/questions.jsonl")
DATA_DIR      = Path("data")

# How many chunks to sample — start with 500, go up to 2000
NUM_CHUNKS = 700

# Delay between API calls (seconds) — be polite to the API
API_DELAY = 0.5


# ── Prompt ────────────────────────────────────────────────────────────────────

QUESTION_PROMPT = """\
You are a Buddhist scholar and practitioner with deep knowledge of the Pali Canon.

Given the following sutta passage, generate exactly 3 questions that a practitioner \
or scholar might ask about it. The questions should range from:
1. Factual — something directly answered in the text
2. Interpretive — requires understanding the teaching's meaning
3. Philosophical/contemplative — broader, deeper, invites reflection

Return ONLY a JSON array of 3 strings. No explanation, no preamble, no markdown.
Example format: ["Question 1?", "Question 2?", "Question 3?"]

Sutta passage ({uid} — {title}):
{text}
"""

# ── API clients ───────────────────────────────────────────────────────────────

def call_deepseek(text: str, uid: str, title: str) -> list[str]:
    """Call DeepSeek API — much cheaper, comparable quality for this task."""
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com"
    )

    prompt = QUESTION_PROMPT.format(uid=uid, title=title, text=text[:2000])

    response = client.chat.completions.create(
        model="deepseek-chat",   # deepseek-chat (V3) — more reliable JSON output than R1 for structured tasks
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.3,
    )

    raw = response.choices[0].message.content
    if raw is None:
        raise ValueError("DeepSeek returned empty content")

    return parse_questions_robust(raw)


def parse_questions_robust(raw: str) -> list[str]:
    """
    Robustly extract a list of 3 questions from messy LLM output.
    Handles: markdown fences, <think> blocks, extra text before/after JSON,
    unterminated strings, single quotes instead of double quotes.
    """
    # 1. Strip <think>...</think> block (DeepSeek-R1 specific)
    if "<think>" in raw:
        if "</think>" in raw:
            raw = raw.split("</think>", 1)[-1]
        else:
            # Unterminated think block — take everything after last </think> or drop
            raw = raw.split("<think>", 1)[0]

    raw = raw.strip()

    # 2. Strip markdown code fences (```json ... ``` or ``` ... ```)
    raw = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()

    # 3. Try direct JSON parse first
    try:
        result = json.loads(raw)
        if isinstance(result, list) and len(result) >= 1:
            questions = [str(q).strip() for q in result if str(q).strip()]
            if len(questions) >= 3:
                return questions[:3]
    except json.JSONDecodeError:
        pass

    # 4. Find the JSON array by locating [ ... ] substring
    start = raw.find("[")
    end   = raw.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidate = raw[start:end+1]
        try:
            result = json.loads(candidate)
            if isinstance(result, list):
                questions = [str(q).strip() for q in result if str(q).strip()]
                if len(questions) >= 3:
                    return questions[:3]
        except json.JSONDecodeError:
            pass

        # 4b. Try fixing common issues: single quotes → double quotes
        try:
            fixed = candidate.replace("'", '"')
            result = json.loads(fixed)
            if isinstance(result, list):
                questions = [str(q).strip() for q in result if str(q).strip()]
                if len(questions) >= 3:
                    return questions[:3]
        except json.JSONDecodeError:
            pass

    # 5. Last resort: extract quoted strings manually using regex
    # Matches both "..." and '...' that look like questions
    patterns = re.findall(r'"([^"]{10,}?)"', raw)
    if len(patterns) >= 3:
        return [p.strip() for p in patterns[:3]]

    patterns = re.findall(r"'([^']{10,}?)'", raw)
    if len(patterns) >= 3:
        return [p.strip() for p in patterns[:3]]

    # 6. Extract numbered lines as fallback (1. ... 2. ... 3. ...)
    lines = [l.strip() for l in raw.split("\n") if l.strip()]
    numbered = []
    for line in lines:
        # Match "1. question" or "1) question" or "- question"
        m = re.match(r"^(?:\d+[.)]\s*|-\s*)(.+)", line)
        if m:
            numbered.append(m.group(1).strip())
    if len(numbered) >= 3:
        return numbered[:3]

    raise ValueError(f"Could not parse 3 questions from response. Raw output was:\n{raw[:300]}")


def generate_questions(text: str, uid: str, title: str) -> list[str]:
    """Route to the correct API."""
    return call_deepseek(text, uid, title)


# ── Chunk loader ──────────────────────────────────────────────────────────────

def load_chunks(path: Path) -> list[dict]:
    """Load your existing JSONL chunks file."""
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # Add a stable chunk_id if not present
            if "chunk_id" not in obj:
                obj["chunk_id"] = i
            chunks.append(obj)
    return chunks


def load_already_done(path: Path) -> set:
    """Resume support — find chunk_ids already processed (strings or ints)."""
    done = set()
    if not path.exists():
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                done.add(str(obj["chunk_id"]))  # always store as string
            except Exception:
                pass
    return done


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    DATA_DIR.mkdir(exist_ok=True)

    provider = "DeepSeek-Chat"
    print(f"Provider: {provider}")

    # Load chunks
    print(f"Loading chunks from {CHUNKS_FILE}...")
    all_chunks = load_chunks(CHUNKS_FILE)
    print(f"Total chunks: {len(all_chunks):,}")

    # Sample — prefer diversity across nikayas if metadata exists
    # Group by nikaya and sample proportionally
    by_nikaya: dict[str, list] = {}
    for c in all_chunks:
        nk = c.get("nikaya", "unknown")
        by_nikaya.setdefault(nk, []).append(c)

    sampled = []
    per_nikaya = max(1, NUM_CHUNKS // max(len(by_nikaya), 1))
    for nk, nk_chunks in by_nikaya.items():
        take = min(per_nikaya, len(nk_chunks))
        sampled.extend(random.sample(nk_chunks, take))

    # Top up to NUM_CHUNKS if needed
    if len(sampled) < NUM_CHUNKS:
        remaining = [c for c in all_chunks if c not in sampled]
        extra = min(NUM_CHUNKS - len(sampled), len(remaining))
        sampled.extend(random.sample(remaining, extra))

    sampled = sampled[:NUM_CHUNKS]
    print(f"   Sampled: {len(sampled):,} chunks across {len(by_nikaya)} nikayas")

    # Resume support
    already_done = load_already_done(QUESTIONS_OUT)
    todo = [c for c in sampled if str(c["chunk_id"]) not in already_done]
    print(f"   Already done: {len(already_done):,} | Remaining: {len(todo):,}")

    if not todo:
        print("All chunks already processed!")
        return

    # Estimate cost
    avg_tokens_per_call = 800  # prompt ~600 + response ~200
    est_cost = (len(todo) * avg_tokens_per_call / 1_000_000) * 0.27  # deepseek-chat pricing
    print(f"   Estimated DeepSeek API cost: ~${est_cost:.2f}")

    # Process
    errors = 0
    with open(QUESTIONS_OUT, "a", encoding="utf-8") as out_f:
        for sutta_chunk in tqdm(todo, desc="Generating questions"):
            cid   = str(sutta_chunk["chunk_id"])
            uid   = sutta_chunk.get("uid", "unknown")
            title = sutta_chunk.get("title", "Unknown Sutta")
            text  = sutta_chunk.get("text") or sutta_chunk.get("chunk_text", "")

            if not text or len(text.split()) < 30:
                continue

            try:
                qs = generate_questions(text, uid, title)

                record = {
                    "chunk_id":   cid,
                    "uid":        uid,
                    "title":      title,
                    "nikaya":     sutta_chunk.get("nikaya", ""),
                    "chunk_text": text,
                    "questions":  qs,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                time.sleep(API_DELAY)

            except ValueError as e:
                errors += 1
                tqdm.write(f"⚠ Parse error on {cid} ({uid}): {e}")
                time.sleep(API_DELAY)

            except Exception as e:
                errors += 1
                err_str = str(e).lower()
                if "rate" in err_str or "429" in err_str:
                    tqdm.write(f"⚠ Rate limit — waiting 30s... ({uid})")
                    time.sleep(30)
                else:
                    tqdm.write(f"⚠ API error on {cid} ({uid}): {e}")
                    time.sleep(API_DELAY * 4)

    total_done = len(already_done) + len(todo) - errors
    print(f"\n✅ Done. Questions saved to {QUESTIONS_OUT}")
    print(f"   Total records: {total_done:,} | Errors skipped: {errors}")
    print(f"\nNext step: python phase1_step2_generate_cot_answers.py")


if __name__ == "__main__":
    main()
