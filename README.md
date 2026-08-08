# quant-research

Research into quantization for reasoning LLMs, focused on how quantization error
accumulates over long chain-of-thought (CoT) generations rather than on
single-forward-pass accuracy.

## Motivation

Standard 4-bit weight quantization (NVFP4 + rotation-based outlier smoothing,
e.g. MR-GPTQ) is now close to lossless for typical LLM workloads. Reasoning
models are the exception: quantization error compounds across long CoT
traces, failures emerge at an identifiable "first vulnerable step" and then
cascade to the final answer, and models sometimes reach the right answer
internally but fail to surface it after quantization. Current mitigations
(inference-time monitoring, targeted re-decoding) patch around this rather
than addressing the root cause. MoE-outlier and VLM-modality quantization are
already crowded, fast-converging lines; this project targets the
comparatively underexplored reasoning/long-CoT regime instead.

## Research plan

1. **Attribute the error source.** Isolate whether cascading failure on
   quantized reasoning traces is driven by weight quantization, KV-cache
   quantization, or activation quantization, rather than assuming it's
   uniform across the stack.
2. **Precision-adaptive decoding.** Use a cheap online signal (token entropy /
   logit margin, in the spirit of speculative decoding's draft-verify split)
   to detect pivot reasoning steps and promote only those to higher
   precision, leaving the bulk of the chain at low bit-width.
3. **Cascade-aware calibration.** Calibrate rotation/reconstruction objectives
   on real CoT traces, weighted by how much a step's error historically
   cascades, instead of generic calibration sets.
4. **Joint weight/KV-cache precision co-design.** Long CoT is KV-cache-bound
   more than weight-bound at scale; treat the two precision budgets jointly
   rather than independently.
5. **Cascade-metric evaluation.** Evaluate with first-error-step and
   error-propagation-rate metrics rather than perplexity or single-number
   final-answer accuracy, since these are what actually predict reasoning
   failure under quantization.

## Status

Stage 0 harness (`cascade.py`) is implemented and unit-tested against synthetic data. Real-model
wiring now exists too: `hf_runner.py` loads Qwen3-4B-Thinking-2507 in BF16/NF4 and produces the
`StepDistribution`s `cascade.py` needs, and `notebooks/stage0_colab.ipynb` runs the full Stage 0
loop against a MATH-500 subset. That notebook hasn't actually been executed yet (needs a Colab GPU
runtime) — pending an actual run.

## Layout

```
src/quant_research/     # library code (cascade.py is pure/model-agnostic; hf_runner.py is the only
                         # module that depends on torch/transformers/bitsandbytes, see `hf` extras)
notebooks/               # Colab notebooks for real-model runs
```
