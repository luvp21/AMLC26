"""Empirically check what this GPU actually supports before the sprint —
don't trust docs/assumptions about Pascal + bitsandbytes, verify directly.
Run after setup_env.sh: `python test_gpu.py`
"""
from __future__ import annotations

import time

import torch


def check_basics():
    print("=== Basics ===")
    print("torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        print("No GPU visible to this process. If using SLURM, check your")
        print("allocation requested a GPU (e.g. --gres=gpu:1).")
        raise SystemExit(1)
    print("device:", torch.cuda.get_device_name(0))
    print("compute capability:", torch.cuda.get_device_capability(0))
    print("total memory (GB):", torch.cuda.get_device_properties(0).total_memory / 1e9)


def check_fp16_matmul():
    print("\n=== fp16 matmul (expected to work, Pascal has native fp16) ===")
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(20):
        c = a @ b
    torch.cuda.synchronize()
    print(f"OK — 20x 4096^2 fp16 matmul took {time.time() - t0:.2f}s")


def check_bf16():
    print("\n=== bf16 support (expected: NOT supported on Pascal) ===")
    supported = torch.cuda.is_bf16_supported()
    print("torch.cuda.is_bf16_supported():", supported)
    if not supported:
        print("Confirmed: use fp16 everywhere on this box, not bf16.")


def check_bnb_4bit():
    print("\n=== bitsandbytes 4-bit quantization (uncertain on compute cap 6.0) ===")
    try:
        from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
        )
        t0 = time.time()
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=2, quantization_config=quant_config, device_map="auto"
        )
        dummy = torch.randint(0, 1000, (2, 32)).to("cuda")
        with torch.no_grad():
            out = model(dummy)
        print(f"OK — 4-bit load + forward pass succeeded in {time.time() - t0:.2f}s. Output shape: {out.logits.shape}")
        print("QLoRA should work on this box.")
    except Exception as e:  # noqa: BLE001 — this is an exploratory diagnostic, not production code
        print(f"FAILED: {e}")
        print("Fall back to plain LoRA in fp16 (no quantization) on this box instead —")
        print("16GB VRAM still fits DeBERTa-base/RoBERTa-large or a 2B-class VLM unquantized.")


if __name__ == "__main__":
    check_basics()
    check_fp16_matmul()
    check_bf16()
    check_bnb_4bit()
