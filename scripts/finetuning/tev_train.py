#!/usr/bin/env python3
"""Local LoRA SFT for Tev1-4B on the legal decision dataset."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import torch
from torch.utils.data import Dataset

MODEL_ID = "togethercomputer/Tev1-4B-experimental"

class TevDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_length: int):
        self.items = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                prompt_ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
                full_ids = tokenizer(row["prompt"] + row["completion"], add_special_tokens=False)["input_ids"]
                if len(full_ids) > max_length:
                    raise ValueError(f"{path}:{line_no}: {len(full_ids)} tokens exceeds max_length={max_length}")
                labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
                self.items.append({
                    "input_ids": full_ids,
                    "attention_mask": [1] * len(full_ids),
                    "labels": labels,
                })
        if not self.items:
            raise ValueError(f"No examples found in {path}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

class Collator:
    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def __call__(self, features):
        max_len = max(len(x["input_ids"]) for x in features)
        input_ids, attention, labels = [], [], []
        for x in features:
            pad = max_len - len(x["input_ids"])
            input_ids.append(x["input_ids"] + [self.pad_id] * pad)
            attention.append(x["attention_mask"] + [0] * pad)
            labels.append(x["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

def load_model(model_id, dtype):
    from transformers import AutoModelForCausalLM
    candidates = []
    try:
        from transformers import AutoModelForImageTextToText
        candidates.append(AutoModelForImageTextToText)
    except ImportError:
        pass
    try:
        from transformers import AutoModelForMultimodalLM
        candidates.append(AutoModelForMultimodalLM)
    except ImportError:
        pass
    candidates.append(AutoModelForCausalLM)

    last_error = None
    for cls in candidates:
        try:
            print(f"Loading model with {cls.__name__}: {model_id}", flush=True)
            model = cls.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map={"": 0},
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            print(f"Loaded with {cls.__name__}", flush=True)
            return model
        except Exception as exc:
            last_error = exc
            print(f"{cls.__name__} failed: {type(exc).__name__}: {exc}", flush=True)
    raise RuntimeError("Could not load Tev1 with available Transformers model classes") from last_error

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--data", type=Path, default=Path("datasets/legal/tev1/instruction"))
    ap.add_argument("--output", type=Path, default=Path("outputs/tev1-legal"))
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=768)
    ap.add_argument("--learning-rate", type=float, default=5e-5)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--save-steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required for this local recipe.")

    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("VRAM GB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2), flush=True)

    from transformers import AutoTokenizer, Trainer, TrainingArguments
    from peft import LoraConfig, TaskType, get_peft_model

    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_path = args.data / "train.jsonl"
    dev_path = args.data / "dev.jsonl"
    if not train_path.is_file() or not dev_path.is_file():
        raise SystemExit("Prepared Tev1 data not found. Run tev_prepare.py first.")

    model = load_model(args.model, torch.float16)
    if hasattr(model, "config"):
        model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules="all-linear",
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = TevDataset(train_path, tokenizer, args.max_length)
    dev_dataset = TevDataset(dev_path, tokenizer, args.max_length)
    steps_per_epoch = math.ceil(len(train_dataset) / (args.batch_size * args.grad_accum))
    print(f"Train={len(train_dataset)} Dev={len(dev_dataset)} optimizer_steps/epoch={steps_per_epoch}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        weight_decay=0.0,
        max_grad_norm=1.0,
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        eval_strategy="steps",
        eval_steps=args.save_steps,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=args.seed,
        optim="adamw_torch",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=Collator(tokenizer),
    )
    print("Starting Tev1 legal fine-tuning...", flush=True)
    trainer.train()

    final_dir = args.output / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    metadata = {
        "base_model": args.model,
        "dataset_train": str(train_path),
        "dataset_dev": str(dev_path),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.grad_accum,
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "target_modules": "all-linear",
        "completion_only": True,
        "seed": args.seed,
    }
    (final_dir / "training_config.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved LoRA adapter: {final_dir}", flush=True)

if __name__ == "__main__":
    main()
