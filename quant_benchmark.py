"""
Quantization-impact investigation for Orthrus-Qwen3-8B.

Background
----------
Orthrus is a dual-view diffusion LM: a frozen autoregressive (AR) head verifies
tokens proposed by a trained diffusion head. The two heads share the embedding,
MLP, and KV cache; only the attention projections and per-head norms are
duplicated as `_diff` variants. The diffusion head was trained to match the AR
head's predictive distribution at bf16. Lowering the AR teacher's precision
shifts that distribution and should reduce per-iteration acceptance length
(tokens-per-forward-pass, TPF).

This script measures four configurations on a fixed prompt set:

  baseline-bf16  - no quantization.
  teacher-int8   - non-diff (AR + shared) weights cast to int8 then back to bf16.
  teacher-int4   - same, but int4. Per-tensor by default; --int4-per-channel
                   switches to per-output-channel scales.
  full-int8      - both AR and diffusion weights cast to int8.

Quantization is *simulated*: weights are cast to a low-precision integer and
immediately dequantized back to bf16. The kernel path is unchanged, so this
isolates the distribution-shift effect from runtime-performance effects (and
therefore reports no memory savings). For each non-baseline config we compare
the generated token sequence against baseline-bf16 (exact match, token-level
Levenshtein, position of first divergence).

Greedy decoding only (do_sample=False, temperature=0.0); divergence is purely
from precision, not sampling RNG.
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
ORTHRUS_REVISION = "34429bd987c2750bed61d65583c6879964367059"

PROMPTS = {
    "short": (
        "Write a program to count the frequency of each word in a paragraph."
    ),
    "long": (
        "Implement a Python class BoundedPriorityQueue backed by a binary heap. "
        "The class takes a capacity (int) and max_heap (bool, default False) at "
        "construction. Implement: push(item, priority) which adds the item and raises "
        "RuntimeError if at capacity; pop() which removes and returns the highest-priority "
        "item, raising IndexError if empty; peek() which returns the best item without "
        "removing it, raising IndexError if empty; __len__; __bool__; __iter__ yielding "
        "(item, priority) pairs in priority order without mutating the queue; and a "
        "classmethod from_items(items, capacity, max_heap=False) accepting an iterable "
        "of (item, priority) pairs. Use full type annotations throughout. Then write a "
        "complete unittest.TestCase covering: push/pop round-trip, capacity enforcement, "
        "min-heap and max-heap ordering, peek, iteration order, from_items bulk loading, "
        "empty-queue edge cases, and duplicate priorities."
    ),
}

CONFIGS = ["baseline-bf16", "teacher-int8", "teacher-int4", "full-int8"]
BASELINE_KEY = "baseline-bf16"

DEFAULT_MAX_NEW_TOKENS = 2048
DEFAULT_WARMUP_TOKENS = 32
DEFAULT_OUTPUT = "results/quant_results.json"


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompts", action="append", metavar="NAME", default=None,
                   help=f"Prompt name(s). May be repeated. Default: all. "
                        f"Valid: {', '.join(PROMPTS)}")
    p.add_argument("--configs", action="append", metavar="NAME", default=None,
                   help=f"Configurations to run. May be repeated. "
                        f"Default: all. Valid: {', '.join(CONFIGS)}")
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    p.add_argument("--warmup-tokens", type=int, default=DEFAULT_WARMUP_TOKENS)
    p.add_argument("--seed", type=int, default=None,
                   help="RNG seed set before each measurement run "
                        "(optional; greedy decoding is already deterministic).")
    p.add_argument("--int4-per-channel", action="store_true", default=False,
                   help="Use per-output-channel symmetric int4 instead of "
                        "per-tensor int4. Less lossy; use as a fallback if "
                        "per-tensor produces unusable output.")
    p.add_argument("--verify-only", action="store_true", default=False,
                   help="Load the model, print the AR / diffusion parameter "
                        "partition for confirmation, then exit without running "
                        "any benchmark.")
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--orthrus-revision", default=ORTHRUS_REVISION)
    args = p.parse_args()

    selected_prompts = args.prompts if args.prompts is not None else list(PROMPTS)
    unknown = [n for n in selected_prompts if n not in PROMPTS]
    if unknown:
        p.error(f"Unknown prompt name(s): {', '.join(unknown)}. "
                f"Valid: {', '.join(PROMPTS)}")
    args.prompts = selected_prompts

    selected_configs = args.configs if args.configs is not None else list(CONFIGS)
    unknown = [n for n in selected_configs if n not in CONFIGS]
    if unknown:
        p.error(f"Unknown config name(s): {', '.join(unknown)}. "
                f"Valid: {', '.join(CONFIGS)}")
    if BASELINE_KEY not in selected_configs and any(c != BASELINE_KEY for c in selected_configs):
        # Non-baseline configs need a baseline to compare against; force-include.
        selected_configs = [BASELINE_KEY] + [c for c in selected_configs if c != BASELINE_KEY]
    args.configs = selected_configs
    return args


# ----------------------------------------------------------------------------
# Model loading and parameter partition
# ----------------------------------------------------------------------------

def load_model(revision):
    print(f"\nLoading {ORTHRUS_ID} (revision={revision}) ...")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        ORTHRUS_ID,
        revision=revision,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(ORTHRUS_ID, revision=revision)
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")
    return model, tokenizer


def partition_params(model):
    """Split parameters into (ar_or_shared, diffusion).

    Diffusion params are those whose qualified name contains '_diff' (the
    `*_proj_diff` and `*_norm_diff` projections added on top of stock Qwen3).
    Everything else (embedding, MLP, layer norms, original Qwen3 attention
    projections, lm_head, final norm) is bundled as 'AR' for this experiment:
    these are the weights the diffusion head was trained to match against.
    """
    ar, diff = [], []
    for name, param in model.named_parameters():
        if "_diff" in name:
            diff.append((name, param))
        else:
            ar.append((name, param))
    return ar, diff


def print_partition(ar, diff):
    ar_n = sum(p.numel() for _, p in ar)
    diff_n = sum(p.numel() for _, p in diff)
    total = ar_n + diff_n
    print(f"\n  AR (non-_diff) parameters:")
    print(f"    tensors: {len(ar):>5}")
    print(f"    numel:   {ar_n:>13,}   ({100 * ar_n / total:5.1f}%)")
    print(f"    sample names:")
    for name, p in ar[:8]:
        print(f"      {name:<60} {tuple(p.shape)}")
    print(f"\n  Diffusion (_diff) parameters:")
    print(f"    tensors: {len(diff):>5}")
    print(f"    numel:   {diff_n:>13,}   ({100 * diff_n / total:5.1f}%)")
    print(f"    sample names:")
    for name, p in diff[:8]:
        print(f"      {name:<60} {tuple(p.shape)}")
    print(f"\n  Total: {total:,} parameters")
    print(f"  Expected from spec: AR ~84%, Diffusion ~16%")


# ----------------------------------------------------------------------------
# Simulated quantization
# ----------------------------------------------------------------------------

@torch.no_grad()
def quantize_int8_per_tensor(param):
    """Symmetric per-tensor int8 quantize-then-dequantize, in place."""
    if param.numel() == 0:
        return
    w = param.data
    max_abs = w.abs().max()
    if max_abs.item() == 0.0:
        return
    scale = (max_abs / 127.0).to(w.dtype)
    q = (w / scale).round().clamp_(-128, 127)
    param.data = (q * scale).to(w.dtype)


@torch.no_grad()
def quantize_int4_per_tensor(param):
    """Symmetric per-tensor int4 (range -8..7) quantize-then-dequantize."""
    if param.numel() == 0:
        return
    w = param.data
    max_abs = w.abs().max()
    if max_abs.item() == 0.0:
        return
    scale = (max_abs / 7.0).to(w.dtype)
    q = (w / scale).round().clamp_(-8, 7)
    param.data = (q * scale).to(w.dtype)


@torch.no_grad()
def quantize_int4_per_channel(param):
    """Symmetric int4 with one scale per output channel (dim 0 of weight).

    Falls back to per-tensor for 1-D params (norms, biases).
    """
    if param.numel() == 0:
        return
    w = param.data
    if w.dim() < 2:
        return quantize_int4_per_tensor(param)
    reduce_dims = tuple(range(1, w.dim()))
    max_abs = w.abs().amax(dim=reduce_dims, keepdim=True)
    scale = (max_abs / 7.0).to(w.dtype)
    # Avoid divide-by-zero on zero rows.
    safe = scale.clone()
    safe[safe == 0] = 1.0
    q = (w / safe).round().clamp_(-8, 7)
    q[max_abs.expand_as(q) == 0] = 0
    param.data = (q * scale).to(w.dtype)


def apply_quantization(model, config, int4_per_channel):
    ar, diff = partition_params(model)
    ar_params = [p for _, p in ar]
    diff_params = [p for _, p in diff]

    if config == "baseline-bf16":
        return
    if config == "teacher-int8":
        for p in ar_params:
            quantize_int8_per_tensor(p)
    elif config == "teacher-int4":
        fn = quantize_int4_per_channel if int4_per_channel else quantize_int4_per_tensor
        for p in ar_params:
            fn(p)
    elif config == "full-int8":
        for p in ar_params + diff_params:
            quantize_int8_per_tensor(p)
    else:
        raise ValueError(f"Unknown config {config!r}")


# ----------------------------------------------------------------------------
# TPF instrumentation
# ----------------------------------------------------------------------------

class PassCounter:
    """Forward-pre-hook counter for diffusion-mode iterations.

    Each Orthrus diffusion step makes one forward call with is_diffusion_pass=True
    (the proposal) followed by one with is_diffusion_pass=False (the AR verify),
    plus a single AR-mode call at generation start. We count diffusion passes;
    TPF = generated_tokens / diffusion_passes is the per-iteration acceptance
    length and is the quantity that should degrade under teacher quantization.
    """

    def __init__(self):
        self.diff_passes = 0
        self.total_passes = 0
        self._handle = None

    def reset(self):
        self.diff_passes = 0
        self.total_passes = 0

    def attach(self, model):
        def pre_hook(_module, _args, kwargs):
            self.total_passes += 1
            if kwargs.get("is_diffusion_pass", False):
                self.diff_passes += 1
        self._handle = model.register_forward_pre_hook(pre_hook, with_kwargs=True)
        return self

    def detach(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


# ----------------------------------------------------------------------------
# Losslessness metrics
# ----------------------------------------------------------------------------

def first_divergence(a, b):
    """Index of first position where token sequences differ, or None if equal."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def levenshtein(a, b):
    """Token-level Levenshtein distance. O(len(a) * len(b)) time, O(len(b)) space."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[m]


# ----------------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------------

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


def _set_seed(seed):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)


@torch.inference_mode()
def generate_diffusion(model, input_ids, max_new_tokens):
    return model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_diffusion_mode=True,
    )


def measure(model, tokenizer, counter, prompt_name, input_ids,
            warmup_tokens, max_new_tokens, seed, label):
    print(f"\n  --- {label}: prompt={prompt_name} ---")

    _set_seed(seed)
    counter.reset()
    print(f"  warmup ({warmup_tokens} tokens) ... "
          "[first call may take several minutes to compile kernels]")
    torch.cuda.synchronize()
    _ = generate_diffusion(model, input_ids, warmup_tokens)
    torch.cuda.synchronize()
    print("  warmup done")

    _set_seed(seed)
    counter.reset()
    torch.cuda.synchronize()
    start = time.perf_counter()
    output_ids = generate_diffusion(model, input_ids, max_new_tokens)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    prompt_len = input_ids.shape[-1]
    new_token_ids = output_ids[0, prompt_len:].tolist()
    new_token_count = len(new_token_ids)
    diff_passes = counter.diff_passes
    total_passes = counter.total_passes
    tpf = new_token_count / diff_passes if diff_passes > 0 else float("nan")
    tps = new_token_count / elapsed if elapsed > 0 else float("nan")

    decoded = tokenizer.decode(new_token_ids, skip_special_tokens=True)
    snippet = decoded[:300].replace("\n", " ")
    print(f"  tokens:        {new_token_count}")
    print(f"  elapsed:       {elapsed:.2f} s")
    print(f"  throughput:    {tps:.1f} tok/s")
    print(f"  diff passes:   {diff_passes}  (total forward calls: {total_passes})")
    print(f"  TPF:           {tpf:.2f}")
    print(f"  output[0:300]: {snippet!r}")

    return {
        "tokens": new_token_count,
        "elapsed": round(elapsed, 3),
        "tps": round(tps, 2),
        "tpf": round(tpf, 3),
        "diffusion_passes": diff_passes,
        "total_forward_passes": total_passes,
        "output_token_ids": new_token_ids,
        "snippet": snippet,
    }


def free_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

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
    print(f"configs:  {', '.join(args.configs)}")
    print(f"prompts:  {', '.join(args.prompts)}")
    int4_mode = "per-channel" if args.int4_per_channel else "per-tensor"
    print(f"int4 mode: {int4_mode}")

    # -----------------------------------------------------------------
    # Verification-only mode
    # -----------------------------------------------------------------
    if args.verify_only:
        print("\n=== Verification mode: parameter partition only ===")
        model, _tok = load_model(args.orthrus_revision)
        ar, diff = partition_params(model)
        print_partition(ar, diff)
        print("\nVerification complete. Re-run without --verify-only to execute "
              "the benchmark.")
        free_model(model)
        return

    # -----------------------------------------------------------------
    # Full benchmark
    # -----------------------------------------------------------------
    # results[config][prompt] = measurement dict
    results = {c: {} for c in args.configs}

    for config in args.configs:
        print(f"\n{'=' * 70}\n=== Configuration: {config} ===\n{'=' * 70}")
        model, tokenizer = load_model(args.orthrus_revision)

        if config == BASELINE_KEY:
            ar, diff = partition_params(model)
            print_partition(ar, diff)

        print(f"\nApplying quantization scheme: {config} "
              f"(int4 mode: {int4_mode}) ...")
        apply_quantization(model, config, args.int4_per_channel)

        counter = PassCounter().attach(model)

        try:
            for prompt_name in args.prompts:
                input_ids = build_inputs(tokenizer, PROMPTS[prompt_name], model.device)
                results[config][prompt_name] = measure(
                    model, tokenizer, counter,
                    prompt_name, input_ids,
                    warmup_tokens=args.warmup_tokens,
                    max_new_tokens=args.max_new_tokens,
                    seed=args.seed,
                    label=config,
                )
        finally:
            counter.detach()

        free_model(model)
        del tokenizer

    # -----------------------------------------------------------------
    # Losslessness comparison vs baseline
    # -----------------------------------------------------------------
    have_baseline = BASELINE_KEY in results and all(
        prompt in results[BASELINE_KEY] for prompt in args.prompts
    )
    if have_baseline:
        print(f"\n{'=' * 70}\n=== Losslessness vs {BASELINE_KEY} ===\n{'=' * 70}")
        for config in args.configs:
            if config == BASELINE_KEY:
                continue
            for prompt_name in args.prompts:
                base = results[BASELINE_KEY][prompt_name]["output_token_ids"]
                this = results[config][prompt_name]["output_token_ids"]
                fdp = first_divergence(base, this)
                exact = (fdp is None)
                # Cap edit distance to keep runtime bounded for catastrophic divergence.
                if exact:
                    edit = 0
                else:
                    edit = levenshtein(base, this)
                base_tpf = results[BASELINE_KEY][prompt_name]["tpf"]
                base_tps = results[BASELINE_KEY][prompt_name]["tps"]
                this_tpf = results[config][prompt_name]["tpf"]
                this_tps = results[config][prompt_name]["tps"]
                tpf_delta = round(this_tpf - base_tpf, 3)
                tps_delta_pct = (
                    round(100.0 * (this_tps - base_tps) / base_tps, 2)
                    if base_tps else None
                )
                vs_base = {
                    "exact_match": exact,
                    "edit_distance": edit,
                    "first_divergence_position": fdp,
                    "tpf_delta": tpf_delta,
                    "throughput_delta_percent": tps_delta_pct,
                }
                results[config][prompt_name]["vs_baseline"] = vs_base
                print(f"  {config:<14} {prompt_name:<6} "
                      f"exact={'yes' if exact else 'no':<3} "
                      f"edit={edit:<5} first_div={fdp}  "
                      f"tpf_delta={tpf_delta:+.2f}  "
                      f"tps_delta={tps_delta_pct:+.1f}%")

    # -----------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------
    print(f"\n{'=' * 70}\n=== Quantization Impact ===\n{'=' * 70}")
    header = (f"  {'config':<14} {'prompt':<6} {'TPF':>6} {'tok/s':>7} "
              f"{'tokens':>7} {'exact':>6} {'first_div':>10}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for config in args.configs:
        for prompt_name in args.prompts:
            r = results[config][prompt_name]
            if config == BASELINE_KEY:
                exact_s, fdiv_s = "---", "---"
            else:
                vb = r.get("vs_baseline", {})
                exact_s = "yes" if vb.get("exact_match") else "no"
                fdiv_s = (str(vb.get("first_divergence_position"))
                          if vb.get("first_divergence_position") is not None
                          else "---")
            print(f"  {config:<14} {prompt_name:<6} "
                  f"{r['tpf']:>6.2f} {r['tps']:>7.1f} {r['tokens']:>7} "
                  f"{exact_s:>6} {fdiv_s:>10}")

    # -----------------------------------------------------------------
    # JSON output
    # -----------------------------------------------------------------
    import accelerate
    import transformers
    cap = torch.cuda.get_device_capability(0)
    mem_bytes = torch.cuda.get_device_properties(0).total_memory

    output = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "hardware": {
            "device_name": torch.cuda.get_device_name(0),
            "compute_capability": list(cap),
            "total_memory_gb": round(mem_bytes / 1024 ** 3, 1),
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
            "prompts": args.prompts,
            "configs": args.configs,
            "max_new_tokens": args.max_new_tokens,
            "warmup_tokens": args.warmup_tokens,
            "seed": args.seed,
            "do_sample": False,
            "int4_mode": int4_mode,
            "orthrus_revision": args.orthrus_revision,
        },
        "notes": {
            "quantization": (
                "Simulated: cast-and-dequantize. Weights round-trip through int8/int4 "
                "but stay in bf16 at runtime. In-memory footprint is unchanged; the "
                "experiment isolates the distribution-shift impact of teacher precision "
                "loss from kernel-performance effects."
            ),
            "tpf": (
                "tokens_per_forward_pass = generated_tokens / number of diffusion-mode "
                "iterations (proposal forward calls). One iteration also performs one AR "
                "verification forward call; total_forward_passes is reported separately."
            ),
            "ar_partition": (
                "AR/teacher = all parameters whose qualified name does NOT contain '_diff' "
                "(embedding, MLP, layer norms, original Qwen3 attention projections, "
                "lm_head, final norm). This is the full set of weights the diffusion head "
                "was trained to match against; quantizing them shifts the target "
                "distribution."
            ),
        },
        "results": results,
    }

    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
