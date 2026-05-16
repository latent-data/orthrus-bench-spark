# Orthrus-Qwen3-8B benchmark on NVIDIA DGX Spark

Reproducible throughput benchmark comparing the Orthrus-Qwen3-8B diffusion LM against a stock Qwen3-8B autoregressive baseline on GB10 (sm_121) hardware.

## TL;DR

| Config | tok/s |
|---|---|
| Orthrus diffusion mode | ~38.9 |
| Orthrus use_diffusion_mode=False | ~3.1 |
| Stock Qwen3-8B AR (KV cache) | TBD (expected 12-18) |
| **Speedup (diffusion vs stock AR)** | **TBD** |

> The Qwen3-8B AR number and final speedup ratio will be filled in after the first complete run on real hardware. The Orthrus numbers above are from initial testing.

## Hardware tested

- **System:** NVIDIA DGX Spark
- **GPU:** NVIDIA GB10, compute capability 12.1 (sm_121), 128 GB unified memory
- **CPU:** aarch64
- **Container:** `nvcr.io/nvidia/pytorch:25.12-py3`
- **torch:** 2.10.0a0+...nv25.12 (custom NGC build, not a PyPI release)
- **flash-attn:** 2.7.x (preinstalled in container)
- **transformers:** 5.8.1
- **accelerate:** 1.13.0

## What this measures and what it does not

**Measured:** single-stream throughput on a fixed prompt with greedy (deterministic) decoding, up to 2048 new tokens.

**Not measured:** output quality, latency under concurrent load, batch throughput, longer contexts, different prompts, sampling strategies.

**Why three numbers and not two:**
Orthrus exposes a `use_diffusion_mode=False` flag, but this is not a clean autoregressive path. The model was trained with bidirectional attention, so it has no KV cache in that mode, and the numbers are not a fair comparison to standard AR decoding. Stock Qwen3-8B with a proper KV cache is the honest baseline. The `use_diffusion_mode=False` result is included for transparency, not as the comparison worth quoting.

## Quick start

```
git clone <this repo>
cd <repo dir>
./run.sh
```

Results appear in `results/results.json`. First run downloads approximately 35 GB of model weights into `~/.cache/huggingface`.

## Configuration

Flags are forwarded from `run.sh` to `benchmark.py`:

| Flag | Default | Description |
|---|---|---|
| `--prompt` | "Write a program..." | Input prompt |
| `--max-new-tokens` | 2048 | Maximum tokens to generate |
| `--warmup-tokens` | 32 | Tokens generated before timing starts |
| `--output` | results/results.json | Path for JSON output |
| `--orthrus-revision` | pinned SHA | HF commit for chiennv/Orthrus-Qwen3-8B |
| `--qwen-revision` | pinned SHA | HF commit for Qwen/Qwen3-8B |

Example:

```
./run.sh --max-new-tokens 512 --output results/short_run.json
```

## Output format

See `results/EXAMPLE_results.json` for the full schema. Top-level fields:

- `timestamp_utc` - ISO 8601 timestamp
- `hardware` - device name, compute capability, memory
- `container` - image tag and name (baked in by Dockerfile)
- `software` - torch, CUDA, transformers, accelerate, flash-attn versions
- `config` - prompt, token limits, decoding settings, pinned revisions
- `results` - per-model: tokens generated, elapsed seconds, tok/s, 300-char output snippet
- `speedup_orthrus_diffusion_vs_qwen3_ar` - ratio of diffusion tok/s to stock AR tok/s

## Reproducibility notes

- **Pinned model revisions:**
  - `chiennv/Orthrus-Qwen3-8B`: `34429bd987c2750bed61d65583c6879964367059`
  - `Qwen/Qwen3-8B`: `b968826d9c46dd6066d109eabc6255188de91218`
- **Pinned container:** `nvcr.io/nvidia/pytorch:25.12-py3`
- **Pinned dependencies:** transformers 5.8.1, accelerate 1.13.0 (see `requirements.txt`)
- **Deterministic decoding:** `do_sample=False` (greedy) for all three configurations
- First run downloads approximately 19 GB (Orthrus) + 16 GB (Qwen3-8B) of weights

## Limitations and caveats

- Targets sm_121 (GB10) on aarch64. Will not run as-is on x86_64 or other Blackwell variants without container adjustments.
- Uses flash-attn 2 only. FA3/FA4 do not yet support sm_121 as of the time this repo was authored.
- Single GPU only.
- 8B model only, no quantisation.

## License

Apache-2.0. See [LICENSE](LICENSE).

## Acknowledgements

- Orthrus paper: [arxiv.org/abs/2605.12825](https://arxiv.org/abs/2605.12825) / [chiennv2000/orthrus](https://github.com/chiennv2000/orthrus)
- Qwen3-8B model card: [huggingface.co/Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
- DGX Spark community guides: [martimramos/dgx-spark-ml-guide](https://github.com/martimramos/dgx-spark-ml-guide), [natolambert/dgx-spark-setup](https://github.com/natolambert/dgx-spark-setup)
