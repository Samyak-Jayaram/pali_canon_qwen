import json
import random
from pathlib import Path

INPUT_FILE = Path("data/cot_dataset_fixed.jsonl")
TRAIN_OUT  = Path("train.jsonl")
EVAL_OUT   = Path("eval_samples.jsonl")

SEED            = 42
EVAL_RATIO      = 0.05
MIN_ANSWER_CHARS   = 80
MIN_THINKING_WORDS = 30 
MIN_ANSWER_WORDS   = 20


SYSTEM_PROMPT = (
    "You are a knowledgeable Buddhist scholar specializing in the Pali Canon. "
    "Answer strictly from the provided sutta context. "
    "Only cite sutta UIDs that appear in the context passages."
)


def format_example(rec):
    """
    Canonical training format. Always rebuilds context from retrieved_chunks
    so the ### Context: block is identical in structure to what inference does.

    Uses rec["question"] (the raw question) as the user turn — NOT rec["prompt"],
    which contains the full COT_USER template including "Available sources: ..."
    header. Using prompt would double-inject the passages into the training example.
    """
    context_parts = []
    if rec.get("retrieved_chunks"):
        for chunk in rec["retrieved_chunks"][:3]:
            text = (chunk.get("text") or "")[:800]
            if not text:
                continue
            # include uid label so the model learns to associate text with uid
            uid   = chunk.get("uid", "")
            title = chunk.get("title", "")
            label = f"[{uid} — {title}]\n" if uid else ""
            context_parts.append(f"{label}{text}")

    context_block = "\n\n".join(context_parts)

    # Use the raw question, not the full prompt template
    question   = (rec.get("question") or "").strip()
    completion = (rec.get("completion") or "").strip()

    # Fallback: reconstruct completion from flat fields if missing
    if not completion:
        thinking = (rec.get("thinking") or "").strip()
        answer   = (rec.get("answer") or "").strip()
        completion = (
            f"<thinking>\n{thinking}\n</thinking>\n<answer>\n{answer}\n</answer>"
            if thinking else answer
        )

    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"### User:\n{question}\n\n"
        f"### Context:\n{context_block}\n\n"
        f"### Assistant:\n{completion}"
    )


def quality_filter(rec):
    # Explicit hallucination flag from the fixed generator
    if rec.get("hallucinated", False):
        return False

    # quality_ok covers both the hallucination check AND the depth check
    # from is_quality_ok() in the generator — respect it
    if not rec.get("quality_ok", True):
        return False

    # Validate the completion itself
    completion = (rec.get("completion") or "").strip()
    if not completion:
        completion = (rec.get("answer") or rec.get("thinking") or "").strip()

    if len(completion) < MIN_ANSWER_CHARS:
        return False

    # Reject duplicate tag wrapping (malformed records)
    if completion.count("<thinking>") > 1 or completion.count("<answer>") > 1:
        return False

    # Answer field word count (flat field, not the full completion)
    answer   = (rec.get("answer") or "").strip()
    thinking = (rec.get("thinking") or "").strip()

    if answer and len(answer.split()) < MIN_ANSWER_WORDS:
        return False

    if thinking and len(thinking.split()) < MIN_THINKING_WORDS:
        return False

    # Must have at least one retrieved chunk with real text
    chunks = rec.get("retrieved_chunks") or []
    if not any(len((c.get("text") or "")) > 50 for c in chunks):
        return False

    return True


# ── load ──────────────────────────────────────────────────────────────────────
records, skipped = [], 0
skip_reasons = {
    "hallucinated": 0,
    "quality_ok_false": 0,
    "short_completion": 0,
    "duplicate_tags": 0,
    "short_answer_or_thinking": 0,
    "no_chunk_text": 0,
}

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        rec = json.loads(line)

        # Track skip reason for diagnostics
        if rec.get("hallucinated", False):
            skip_reasons["hallucinated"] += 1
            skipped += 1
            continue
        if not rec.get("quality_ok", True):
            skip_reasons["quality_ok_false"] += 1
            skipped += 1
            continue
        if not quality_filter(rec):
            # figure out which sub-check failed
            completion = (rec.get("completion") or rec.get("answer") or "").strip()
            if len(completion) < MIN_ANSWER_CHARS:
                skip_reasons["short_completion"] += 1
            elif completion.count("<thinking>") > 1 or completion.count("<answer>") > 1:
                skip_reasons["duplicate_tags"] += 1
            elif not any(len((c.get("text") or "")) > 50 for c in (rec.get("retrieved_chunks") or [])):
                skip_reasons["no_chunk_text"] += 1
            else:
                skip_reasons["short_answer_or_thinking"] += 1
            skipped += 1
            continue

        records.append(rec)

print(f"Loaded: {len(records):,} | Skipped: {skipped:,}")
print("Skip breakdown:")
for reason, count in skip_reasons.items():
    if count:
        print(f"  {reason:30s}: {count:,}")

if not records:
    raise ValueError("No valid records after filtering.")

# ── split ─────────────────────────────────────────────────────────────────────
random.seed(SEED)
random.shuffle(records)

eval_size  = max(100, int(len(records) * EVAL_RATIO))
eval_recs  = records[:eval_size]
train_recs = records[eval_size:]
print(f"\nTrain: {len(train_recs):,} | Eval: {len(eval_recs):,}")

# ── save train ────────────────────────────────────────────────────────────────
with open(TRAIN_OUT, "w", encoding="utf-8") as f:
    for r in train_recs:
        json.dump({"text": format_example(r)}, f, ensure_ascii=False)
        f.write("\n")

# ── save eval — raw records, no pre-formatting ────────────────────────────────
with open(EVAL_OUT, "w", encoding="utf-8") as f:
    for r in eval_recs:
        json.dump(r, f, ensure_ascii=False)
        f.write("\n")

print(f"\nSaved:\n  Train → {TRAIN_OUT}\n  Eval  → {EVAL_OUT}")

# ── quick sanity check on a sample training example ──────────────────────────
print("\n── Sample training example ──────────────────────────────────────────")
sample = format_example(train_recs[0])
print(sample[:800])
print("..." if len(sample) > 800 else "")