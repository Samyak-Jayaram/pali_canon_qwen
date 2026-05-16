import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

# ── config ────────────────────────────────────────────────────────────────────
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

MAX_SEQ_LEN = 1024
TRAIN_PATH  = "pali-canon-cot-qa/train_v3.jsonl"
EVAL_PATH   = "pali-canon-cot-qa/eval_samples_v3.jsonl"
OUTPUT_DIR  = "/kaggle/working/qwen0.5-adapter"
MERGED_DIR  = "/kaggle/working/qwen10.5-merged"

NUM_EPOCHS  = 3
BATCH_SIZE  = 1
GRAD_ACCUM  = 4
LR          = 2e-4

SYSTEM_PROMPT = (
    "You are a knowledgeable Buddhist scholar specializing in the Pali Canon. "
    "Answer strictly from the provided sutta context. "
    "Only cite sutta UIDs that appear in the context passages."
)

# ── canonical formatter ──────────────
def format_example(rec):
    context_parts = []
    if rec.get("retrieved_chunks"):
        for chunk in rec["retrieved_chunks"][:3]:
            text = (chunk.get("text") or "")[:800]
            if not text:
                continue
            # UID label teaches the model to associate text with citation
            # ADDED vs original — matches split_dataset.py
            uid   = chunk.get("uid", "")
            title = chunk.get("title", "")
            label = f"[{uid} — {title}]\n" if uid else ""
            context_parts.append(f"{label}{text}")

    context_block = "\n\n".join(context_parts)

    # Use raw question which contains the full COT_USER
    # template and would double-inject passages into the training example
    question   = (rec.get("question") or "").strip()
    completion = (rec.get("completion") or "").strip()

    # Fallback: reconstruct from flat fields if completion missing
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


# ── tokenizer ─────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.model_max_length = MAX_SEQ_LEN
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ── data ──────────────────────────────────────────────────────────────────────
# train.jsonl: already {"text": ...} from split_dataset.py — load directly
def load_train_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return Dataset.from_list(rows)

# eval.jsonl: raw records
def load_eval_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                rows.append({"text": format_example(rec)})
    return Dataset.from_list(rows)

train_dataset = load_train_jsonl(TRAIN_PATH)
eval_dataset  = load_eval_jsonl(EVAL_PATH)

def tokenize(ex):
    return tokenizer(
        ex["text"],
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding=False
    )

train_dataset = train_dataset.map(tokenize, remove_columns=["text"])
eval_dataset  = eval_dataset.map(tokenize, remove_columns=["text"])

# ── model ─────────────────────────────────────────────────────────────────────
def load_model():
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb,
        device_map=None,
        torch_dtype=torch.bfloat16,
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    return model

def apply_lora(model):
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,  
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    return get_peft_model(model, config)

# ── train ─────────────────────────────────────────────────────────────────────
def main():
    model = load_model()
    model = apply_lora(model)
    model.print_trainable_parameters()

    args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=False,
        bf16=True,
        logging_steps=20,
        save_steps=50,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=50,
        max_length=MAX_SEQ_LEN,
        packing=False,
        dataloader_num_workers=2,
        report_to="none",
        ddp_find_unused_parameters=False,
        prediction_loss_only=True,
        load_best_model_at_end=True,  
        metric_for_best_model="loss",
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)

    merged = model.merge_and_unload()
    merged.save_pretrained(MERGED_DIR)
    tokenizer.save_pretrained(MERGED_DIR)
    print("DONE")

if __name__ == "__main__":
    main()