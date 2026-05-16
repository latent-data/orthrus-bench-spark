"""
Benchmark Orthrus-Qwen3-8B against a fair baseline.

Compares three settings on the same prompt:
  1. Orthrus, diffusion mode      (use_diffusion_mode=True)
  2. Orthrus, "AR" mode           (use_diffusion_mode=False)
     - probably not a true AR path, included for transparency
  3. Stock Qwen/Qwen3-8B          (proper AR with KV cache, FA2)
     - this is the real baseline worth quoting

Models are loaded one at a time and freed in between to keep memory sane,
though on DGX Spark's 128 GB unified pool you could hold them concurrently.

Note: stock Qwen3-8B is ~16 GB and will download on first run.
"""
import argparse
import datetime
import gc
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ORTHRUS_ID = "chiennv/Orthrus-Qwen3-8B"
QWEN_ID = "Qwen/Qwen3-8B"

# Pinned revisions for reproducibility; override with --orthrus-revision / --qwen-revision
ORTHRUS_REVISION = "34429bd987c2750bed61d65583c6879964367059"
QWEN_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"

DEFAULT_PROMPT = "Write a program to count the frequency of each word in a paragraph."
DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_WARMUP_TOKENS = 32
DEFAULT_OUTPUT = "results/results.json"


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark Orthrus vs Qwen3-8B AR")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    p.add_argument("--warmup-tokens", type=int, default=DEFAULT_WARMUP_TOKENS)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--orthrus-revision", default=ORTHRUS_REVISION)
    p.add_argument("--qwen-revision", default=QWEN_REVISION)
    return p.parse_args()


def load(model_id, revision, trust_remote_code=False):
    print(f"\nLoading {model_id} (revision={revision}) ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
        trust_remote_code=trust_remote_code,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    return model, tokenizer


def build_inputs(tokenizer, prompt, device):
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": prompt},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False,
    ).input_ids
    return input_ids.to(device)


@torch.inference_mode()
def generate(model, input_ids, max_new_tokens, **extra):
    return model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        **extra,
    )


def benchmark(model, tokenizer, input_ids, label, warmup_tokens, max_new_tokens, **extra):
    print(f"\n=== {label} ===")

    print(f"  warmup ({warmup_tokens} tokens) ...")
    _ = generate(model, input_ids, warmup_tokens, **extra)
    torch.cuda.synchronize()

    print(f"  measuring (up to {max_new_tokens} tokens) ...")
    torch.cuda.synchronize()
    start = time.perf_counter()
    output_ids = generate(model, input_ids, max_new_tokens, **extra)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    new_tokens = output_ids.shape[-1] - input_ids.shape[-1]
    tps = new_tokens / elapsed

    print(f"  tokens:     {new_tokens}")
    print(f"  elapsed:    {elapsed:.2f} s")
    print(f"  throughput: {tps:.1f} tok/s")

    response = tokenizer.decode(
        output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True
    )
    snippet = response[:300].replace("\n", " ")
    print(f"  output[0:300]: {snippet!r}")

    return {"tokens": new_tokens, "elapsed": elapsed, "tps": tps, "snippet": snippet}


def free_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _flash_attn_version():
    try:
        import flash_attn
        return flash_attn.__version__
    except ImportError:
        return None


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available; this benchmark assumes a GPU.")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"torch: {torch.__version__}  CUDA: {torch.version.cuda}")
    print(f"compute capability: {torch.cuda.get_device_capability(0)}")

    results = {}

    # ---- Orthrus (diffusion + "AR" toggle)
    model, tokenizer = load(ORTHRUS_ID, args.orthrus_revision, trust_remote_code=True)
    input_ids = build_inputs(tokenizer, args.prompt, model.device)

    results["orthrus_diffusion"] = benchmark(
        model, tokenizer, input_ids,
        "Orthrus diffusion mode",
        warmup_tokens=args.warmup_tokens,
        max_new_tokens=args.max_new_tokens,
        use_diffusion_mode=True,
        do_sample=False,
    )
    results["orthrus_nodiff"] = benchmark(
        model, tokenizer, input_ids,
        "Orthrus use_diffusion_mode=False (likely not true AR)",
        warmup_tokens=args.warmup_tokens,
        max_new_tokens=args.max_new_tokens,
        use_diffusion_mode=False,
        do_sample=False,
    )

    free_model(model)
    del tokenizer, input_ids

    # ---- Stock Qwen3-8B as the real AR baseline
    model, tokenizer = load(QWEN_ID, args.qwen_revision, trust_remote_code=False)
    input_ids = build_inputs(tokenizer, args.prompt, model.device)

    results["qwen3_8b_ar"] = benchmark(
        model, tokenizer, input_ids,
        "Stock Qwen3-8B, standard AR with KV cache",
        warmup_tokens=args.warmup_tokens,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )

    free_model(model)

    # ---- Summary
    print("\n=== Summary ===")
    label_map = {
        "orthrus_diffusion": "Orthrus diffusion",
        "orthrus_nodiff": "Orthrus use_diffusion_mode=False",
        "qwen3_8b_ar": "Qwen3-8B AR (KV cache)",
    }
    label_w = max(len(v) for v in label_map.values()) + 2
    header = f"  {'config':<{label_w}} {'tokens':>8}  {'elapsed':>10}  {'throughput':>12}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for key, r in results.items():
        print(
            f"  {label_map[key]:<{label_w}} "
            f"{r['tokens']:>8}  "
            f"{r['elapsed']:>8.2f} s  "
            f"{r['tps']:>8.1f} tok/s"
        )

    baseline = results["qwen3_8b_ar"]["tps"]
    diff = results["orthrus_diffusion"]["tps"]
    speedup = diff / baseline
    print(f"\n  Orthrus diffusion vs stock Qwen3-8B AR: {speedup:.2f}x")
    print("  (this is the comparison worth quoting)")

    # ---- JSON output
    import accelerate
    import transformers

    cap = torch.cuda.get_device_capability(0)
    mem_bytes = torch.cuda.get_device_properties(0).total_memory

    output = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hardware": {
            "device_name": torch.cuda.get_device_name(0),
            "compute_capability": list(cap),
            "total_memory_gb": round(mem_bytes / 1024**3, 1),
        },
        "container": {
            "image_tag": os.environ.get("BENCHMARK_IMAGE_TAG", "unknown"),
            "image_name": os.environ.get("BENCHMARK_IMAGE_NAME", "unknown"),
        },
        "software": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "accelerate": accelerate.__version__,
            "flash_attn": _flash_attn_version(),
        },
        "config": {
            "prompt": args.prompt,
            "max_new_tokens": args.max_new_tokens,
            "warmup_tokens": args.warmup_tokens,
            "do_sample": False,
            "orthrus_revision": args.orthrus_revision,
            "qwen_revision": args.qwen_revision,
        },
        "results": results,
        "speedup_orthrus_diffusion_vs_qwen3_ar": round(speedup, 2),
    }

    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
