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
import gc
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ORTHRUS_ID = "chiennv/Orthrus-Qwen3-8B"
QWEN_ID = "Qwen/Qwen3-8B"
PROMPT = "Write a program to count the frequency of each word in a paragraph."
MAX_NEW_TOKENS = 2048
WARMUP_TOKENS = 32


def load(model_id, trust_remote_code=False):
    print(f"\nLoading {model_id} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
        trust_remote_code=trust_remote_code,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer


def build_inputs(tokenizer, device):
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": PROMPT},
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


def benchmark(model, tokenizer, input_ids, label, **extra):
    print(f"\n=== {label} ===")

    print(f"  warmup ({WARMUP_TOKENS} tokens) ...")
    _ = generate(model, input_ids, WARMUP_TOKENS, **extra)
    torch.cuda.synchronize()

    print(f"  measuring (up to {MAX_NEW_TOKENS} tokens) ...")
    torch.cuda.synchronize()
    start = time.perf_counter()
    output_ids = generate(model, input_ids, MAX_NEW_TOKENS, **extra)
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

    return tps


def free_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available; this benchmark assumes a GPU.")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"torch: {torch.__version__}  CUDA: {torch.version.cuda}")
    print(f"compute capability: {torch.cuda.get_device_capability(0)}")

    results = {}

    # ---- Orthrus (diffusion + "AR" toggle)
    model, tokenizer = load(ORTHRUS_ID, trust_remote_code=True)
    input_ids = build_inputs(tokenizer, model.device)

    results["Orthrus diffusion"] = benchmark(
        model, tokenizer, input_ids,
        "Orthrus diffusion mode",
        use_diffusion_mode=True,
    )
    results["Orthrus use_diffusion_mode=False"] = benchmark(
        model, tokenizer, input_ids,
        "Orthrus use_diffusion_mode=False (likely not true AR)",
        use_diffusion_mode=False,
    )

    free_model(model)
    del tokenizer, input_ids

    # ---- Stock Qwen3-8B as the real AR baseline
    model, tokenizer = load(QWEN_ID, trust_remote_code=False)
    input_ids = build_inputs(tokenizer, model.device)

    results["Qwen3-8B AR (KV cache)"] = benchmark(
        model, tokenizer, input_ids,
        "Stock Qwen3-8B, standard AR with KV cache",
    )

    free_model(model)

    # ---- Summary
    print("\n=== Summary ===")
    width = max(len(k) for k in results) + 2
    for label, tps in results.items():
        print(f"  {label:<{width}} {tps:6.1f} tok/s")

    baseline = results["Qwen3-8B AR (KV cache)"]
    diff = results["Orthrus diffusion"]
    print(f"\n  Orthrus diffusion vs stock Qwen3-8B AR: {diff / baseline:.2f}x")
    print("  (this is the comparison worth quoting)")


if __name__ == "__main__":
    main()
