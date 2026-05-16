import json
from pathlib import Path
import re

CITATION_RE = re.compile(r'\b(MN|SN|AN|DN)\s*(\d+(?:\.\d+)?)')

with open("data/cot_dataset_fixed.jsonl") as f:
    records = [json.loads(l) for l in f if l.strip()]

hallucinated = 0
for rec in records:
    answer = rec.get("answer", "")
    retrieved_uids = " ".join(
        r.get("uid", "") for r in rec.get("retrieved_chunks", [])
    )
    citations = CITATION_RE.findall(answer)
    for book, num in citations:
        cited = f"{book.lower()}{num}".replace(".", "")
        # check if cited uid appears anywhere in retrieved uids
        if cited not in retrieved_uids.lower():
            hallucinated += 1
            break  # one hallucinated citation per record is enough to flag it

print(f"Records with likely hallucinated citations: {hallucinated}/{len(records)}")
print(f"That's {100*hallucinated/len(records):.1f}% of your training data")


BAD_PHRASES = [
    "human resources", "payroll", "performance evaluation software",
    "management system", "human intelligence consists", "sensory input",
    "cognitive processing"
]
with open("data/cot_dataset_fixed.jsonl") as f:
    for i, line in enumerate(f):
        rec = json.loads(line)
        text = (rec.get("completion","") + rec.get("answer","")).lower()
        if any(p in text for p in BAD_PHRASES):
            print(f"Line {i}: CONTAMINATED — {text[:100]}")