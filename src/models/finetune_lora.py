"""QLoRA fine-tuning for a text encoder (BERT/RoBERTa/DeBERTa-class) doing
regression or classification on catalog text — the 2023 and (text-tower-of-)
2025 pattern. Run this on the AWS GPU box, not locally (see infra/aws/).

For an image/VLM task like 2024's entity extraction, don't reinvent this:
the 2024 winners fine-tuned Qwen2-VL-7B via LLaMA-Factory
(https://github.com/hiyouga/LLaMA-Factory) with QLoRA — it already handles
vision-language SFT end to end. Install it on the GPU box and point it at a
JSON dataset of {image, instruction, output} triples built from
`src/data/loaders.py`. Only fall back to a hand-rolled training loop here if
LLaMA-Factory doesn't fit the announced task shape.

Usage:
    python -m src.models.finetune_lora \
        --model_name microsoft/deberta-v3-base \
        --train_csv data/train.csv --text_col catalog_text --target_col price \
        --task regression --output_dir runs/deberta-lora --dtype bf16

On PARAM Shavak, use `--dtype fp16 --no_4bit` — this is the sprint default there
(see infra/param_shavak/README.md's live-verified caveats, 24 Sept): fp16 is the
only dtype actually matmul-confirmed working on the GP100 (bf16 reports as
"supported" but that's an unverified flag, not a proven fast path), and 4-bit
QLoRA support was never confirmed either way (the test got blocked by a
network issue, not a GPU failure) — `--no_4bit` is the known-safe path. Use
`--dtype bf16` (default 4-bit QLoRA on) on the AWS g5.xlarge (A10G) instead.

Model weights must be pre-downloaded on your laptop and rsynced in — PARAM
Shavak's own connection to HF's CDN gets reset on large binary transfers (see
infra/param_shavak/README.md). Run with `HF_HUB_OFFLINE=1` there once the
weights are in place.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def build_model_and_tokenizer(
    model_name: str, task: str, compute_dtype, num_labels: int = 1, load_in_4bit: bool = True
):
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    problem_type = "regression" if task == "regression" else "single_label_classification"

    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type=problem_type,
            quantization_config=quant_config,
            device_map="auto",
        )
    else:
        # Plain LoRA, no quantization — the fallback path for GPUs (e.g. PARAM
        # Shavak's GP100) where 4-bit bitsandbytes support is unreliable.
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            problem_type=problem_type,
            torch_dtype=compute_dtype,
            device_map="auto",
        )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["query_proj", "value_proj", "query", "value"],  # covers BERT- and DeBERTa-style naming
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="microsoft/deberta-v3-base")
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--text_col", default="catalog_text")
    parser.add_argument("--target_col", required=True)
    parser.add_argument("--task", choices=["regression", "classification"], default="regression")
    parser.add_argument("--output_dir", default="runs/lora-run")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument(
        "--dtype", choices=["bf16", "fp16"], default="bf16",
        help="bf16 on AWS g5.xlarge (A10G); fp16 on PARAM Shavak (Pascal GP100 has no bf16)",
    )
    parser.add_argument(
        "--no_4bit", dest="load_in_4bit", action="store_false",
        help="disable bitsandbytes 4-bit quantization (fallback for GPUs where QLoRA is unreliable, e.g. PARAM Shavak's GP100 — test with infra/param_shavak/test_gpu.py first)",
    )
    args = parser.parse_args()

    import torch
    from datasets import Dataset
    from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

    compute_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    df = pd.read_csv(args.train_csv)
    num_labels = 1 if args.task == "regression" else df[args.target_col].nunique()
    model, tokenizer = build_model_and_tokenizer(
        args.model_name, args.task, compute_dtype, num_labels, load_in_4bit=args.load_in_4bit
    )

    def tokenize(batch):
        enc = tokenizer(batch[args.text_col], truncation=True, max_length=args.max_length)
        if args.task == "regression":
            enc["labels"] = [float(v) for v in batch[args.target_col]]
        else:
            enc["labels"] = batch[args.target_col]
        return enc

    dataset = Dataset.from_pandas(df[[args.text_col, args.target_col]]).train_test_split(test_size=0.1, seed=42)
    dataset = dataset.map(tokenize, batched=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        bf16=(args.dtype == "bf16"),
        fp16=(args.dtype == "fp16"),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=20,
        load_best_model_at_end=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"saved LoRA adapter + tokenizer to {args.output_dir}")


if __name__ == "__main__":
    main()
