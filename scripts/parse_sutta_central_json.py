import json
import re
from pathlib import Path
from tqdm import tqdm

# ── Config ───────────────────────────────────────────────────────────────────

BILARA_DIR  = Path("bilara-data/translation/en/sujato/sutta")
DATA_DIR    = Path("data")
OUTPUT_FILE = DATA_DIR / "chunks.jsonl"

CHUNK_SIZE    = 350   # words per chunk
CHUNK_OVERLAP = 50    # overlap between consecutive chunks

NIKAYAS = {
    "dn": "Dīgha Nikāya (Long Discourses)",
    "mn": "Majjhima Nikāya (Middle Length Discourses)",
    "sn": "Saṃyutta Nikāya (Connected Discourses)",
    "an": "Aṅguttara Nikāya (Numerical Discourses)",
}

# ── Segment key sorting ───────────────────────────────────────────────────────

def segment_sort_key(key: str) -> list[int]:
    """
    Bilara segment keys look like  "mn2:1.3"  or  "an10.98:1.10"
    We sort numerically on all digit groups after stripping the prefix.
    """
    digits = re.findall(r"\d+", key)
    return [int(d) for d in digits]


# ── Title extraction ──────────────────────────────────────────────────────────

def extract_title(segments: dict) -> str:
    """
    The first segment of every sutta is its title (key ends in ':0.1' or ':0.2').
    Fall back to the filename UID if nothing found.
    """
    for key in sorted(segments.keys(), key=segment_sort_key):
        val = segments[key].strip()
        if val and not val.startswith("♦"):
            return val
    return ""


# ── Text assembly ─────────────────────────────────────────────────────────────

def assemble_text(segments: dict) -> str:
    """
    Join all non-empty segment values in reading order.
    Skip:
      - the title segment (:0.1 / :0.2) — already captured separately
      - segments that are only HTML entities or punctuation markers (♦)
      - empty strings
    """
    ordered = sorted(segments.items(), key=lambda x: segment_sort_key(x[0]))
    parts = []
    for i, (key, val) in enumerate(ordered):
        val = val.strip()
        if not val or val.startswith("♦"):
            continue
        if i == 0:
            # Skip title segment
            continue
        parts.append(val)
    return " ".join(parts)


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_text(text: str, meta: dict) -> list[dict]:
    """Split text into overlapping word-count chunks, each carrying full metadata."""
    words = text.split()
    if not words:
        return []

    chunks = []
    i = 0
    chunk_idx = 0

    while i < len(words):
        chunk_words = words[i : i + CHUNK_SIZE]
        chunks.append({
            "text":            " ".join(chunk_words),
            "chunk_id":        f"{meta['uid']}__chunk_{chunk_idx}",
            "chunk_index":     chunk_idx,
            "uid":             meta["uid"],
            "title":           meta["title"],
            "nikaya":          meta["nikaya"],
            "nikaya_name":     meta["nikaya_name"],
            "suttacentral_url": f"https://suttacentral.net/{meta['uid']}/en/sujato",
        })
        i += CHUNK_SIZE - CHUNK_OVERLAP
        chunk_idx += 1

    return chunks


# ── UID from filename ─────────────────────────────────────────────────────────

def uid_from_path(path: Path) -> str:
    """
    Bilara filenames look like:
      mn2_translation-en-sujato.json   →  uid = "mn2"
      an10.98_translation-en-sujato.json  →  uid = "an10.98"
    """
    return path.name.split("_")[0]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    DATA_DIR.mkdir(exist_ok=True)

    if not BILARA_DIR.exists():
        print(f"ERROR: {BILARA_DIR} not found.")
        print("Please clone bilara-data first — see the docstring at the top of this file.")
        return

    all_chunks = []
    total_suttas = 0
    total_skipped = 0

    for nikaya_id, nikaya_name in NIKAYAS.items():
        nikaya_dir = BILARA_DIR / nikaya_id
        if not nikaya_dir.exists():
            print(f"⚠  Skipping {nikaya_id}: directory not found at {nikaya_dir}")
            continue

        # Collect all JSON files recursively (some nikayas have subdirs)
        files = sorted(nikaya_dir.rglob("*.json"))
        print(f"\n📖 {nikaya_name}: {len(files)} files")

        for fpath in tqdm(files, desc=f"  {nikaya_id.upper()}"):
            uid = uid_from_path(fpath)

            try:
                with open(fpath, encoding="utf-8") as f:
                    segments: dict = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                total_skipped += 1
                continue

            if not isinstance(segments, dict) or not segments:
                total_skipped += 1
                continue

            title = extract_title(segments)
            text  = assemble_text(segments)

            if len(text.split()) < 30:
                # Too short to be useful (stubs, single-verse texts, etc.)
                total_skipped += 1
                continue

            meta = {
                "uid":        uid,
                "title":      title or uid.upper(),
                "nikaya":     nikaya_id,
                "nikaya_name": nikaya_name,
            }

            chunks = chunk_text(text, meta)
            all_chunks.extend(chunks)
            total_suttas += 1

    # Write output
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"\n{'='*50}")
    print(f"Done!")
    print(f"    Suttas parsed  : {total_suttas:,}")
    print(f"    Suttas skipped : {total_skipped:,}  (stubs / empty)")
    print(f"    Total chunks   : {len(all_chunks):,}")
    print(f"    Output         : {OUTPUT_FILE}")
    print(f"\n    Average chunks per sutta: {len(all_chunks)/max(total_suttas,1):.1f}")
    print(f"\nNext: python scripts/02_embed_and_index.py")


if __name__ == "__main__":
    main()
