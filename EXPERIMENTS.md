# Experiment Plan

This lays out a staged, executable path from the research plan in README.md to
concrete results. Each stage produces something usable on its own (a harness,
a benchmark, a checkpoint) even if later stages stall.

## Stage 0 — Cascade evaluation harness (prerequisite for everything below)

Before any quantization experiment, we need a way to measure *where* a
reasoning trace goes wrong, not just whether the final answer is wrong.

- Instrument generation to log per-step entropy / logit margin against a
  BF16 reference trace (teacher-forced replay: feed the quantized model's
  prefix, compare its next-token distribution to the BF16 model's at the
  same position).
- Define **first-error-step**: first position where the quantized model's
  argmax token diverges from the BF16 reference in a way that changes the
  downstream derivation (not just a benign paraphrase).
- Define **error-propagation rate**: fraction of divergences at step *i*
  that flip the final answer, as a function of *i*'s position in the trace
  (early vs. late steps).
- Ship as a standalone eval harness — usable to score any quantization
  method (AWQ, GPTQ, MR-GPTQ, NVFP4, SmoothQuant) even outside this project.

**Models:** Qwen3-4B-Thinking, Qwen3-8B (reasoning mode) to start — cheap
enough to iterate on. **Benchmarks:** AIME 2025/2026, MATH-500,
GPQA-Diamond, LiveCodeBench. **Compute:** inference-only, single node.

**Status (2026-08-08):** real-model wiring shipped — `src/quant_research/hf_runner.py` +
`notebooks/stage0_colab.ipynb` run this stage against Qwen3-4B-Thinking-2507 (BF16 reference vs.
bitsandbytes NF4 candidate) on a small MATH-500 subset. Note NF4 is a stand-in for MR-GPTQ/NVFP4
here — Colab hardware has no native FP4 tensor cores, so this is a harness correctness check, not
yet a result usable for Stage 1's attribution claim.

First real execution (same day) surfaced two harness bugs rather than a usable result: the
self-check (BF16 teacher-forced on its own trace) diverged on 8/8 problems, and every trace hit the
1024-token cap before finishing. Fixed: `load_model` now defaults to `attn_implementation="eager"`
(the likely cause was `generate()`'s cached decoding and the harness's bulk forward pass hitting
different fused-attention kernels under sdpa/flash, diverging slightly in bf16); `MAX_NEW_TOKENS`
raised to 4096 with truncation now tracked and reported per-trace and in aggregate.

**Status (2026-08-10): first clean, validated result.** Four more real-Colab bugs surfaced and were
fixed after the above (nested-clone from a non-idempotent clone cell, stale `sys.modules` cache
surviving a code fix within one kernel session, and CUDA OOM — first from concurrent dual-model
residency, then from per-problem allocator fragmentation even with one model resident; see commit
history for `hf_runner.py` and the notebook). With all of those fixed, an 8-problem MATH-500 run
completed end-to-end with no infra errors. Result: candidate (NF4) diverges from the BF16 reference
earlier than the harness's own self-check noise floor in 7/8 problems (2.3×-39× earlier, one exact
tie), and that divergence flips the final answer in 5/8 (62.5%). Self-check's divergence *margin* is
an exact 0.0-nat tie in 5/8 cases (consistent with harmless bf16 rounding noise between incremental
and bulk forward paths) while candidate's margins are typically much larger (up to 2.5 nats) — a
confident distribution shift, not a coin-flip, which is what makes the early-divergence pattern read
as a real quantization effect. n=8 is a smoke-test sample, not a claim; 3/8 traces still hit the
4096-token cap (truncation doesn't affect the divergence-position metric itself, only
`reference_correct`/`candidate_answer` for those specific rows).

Scaling up now: added AIME 2025 alongside MATH-500 (`N_MATH=25`, `N_AIME=15`), with results broken
out per-source since the two benchmarks differ enough in difficulty/length that a pooled rate would
hide real differences. GPQA-Diamond and LiveCodeBench remain deferred — GPQA-Diamond
(`Idavidrein/gpqa`) is gated and needs `huggingface-cli login`; LiveCodeBench needs
code-execution-based correctness checking instead of boxed-answer extraction. Both are a genuinely
different scaffolding problem, not just more data.

## Stage 1 — Error attribution (weight vs. KV-cache vs. activation)

**Hypothesis:** cascading failure is not uniform across quantization axes —
one of {weight, KV-cache, activation} quantization dominates the
first-error-step distribution.

**Method:** factorial ablation at matched bit-width (e.g. 4-bit), holding
the other two axes at BF16:
1. BF16 baseline (all axes full precision)
2. Weights → W4 (MR-GPTQ / NVFP4) only
3. KV-cache → 4-bit only
4. Activations → A4 only
5. All three combined (current standard practice)

Score each condition with the Stage 0 harness. Compare first-error-step
position distributions and error-propagation rates across conditions, not
just final accuracy.

**Success criterion:** a clear ranking of which axis drives cascade risk,
directly motivating where Stages 2–4 should spend engineering effort.

## Stage 2 — Cascade-aware calibration

**Hypothesis:** calibrating quantization (GPTQ/MR-GPTQ reconstruction
objective) on generic text under-weights the layers that matter most for
reasoning-trace cascades; reweighting calibration by Stage 1's per-layer
cascade contribution recovers accuracy without any runtime cost.

**Method:** compute a per-layer sensitivity score from Stage 1's ablations
(how often perturbing this layer correlates with a first-error-step), use
it to weight the calibration reconstruction loss. Compare against:
- Standard calibration on generic text (WikiText/C4)
- Standard calibration on reasoning traces, unweighted
- Cascade-weighted calibration (proposed)

**Metric:** final accuracy on AIME/MATH-500/GPQA-Diamond at fixed bit-width,
zero runtime overhead (this stage only touches the offline calibration
step). This is the cheapest stage to try, reusing existing GPTQ tooling —
run it in parallel with Stage 3, not strictly after it.

## Stage 3 — Precision-adaptive decoding (core contribution)

**Hypothesis:** promoting only detected pivot steps to higher precision
recovers most BF16 accuracy at a small fraction of the cost of uniform
higher-precision quantization.

**Method:**
1. Calibrate a pivot-step detector (entropy / logit-margin threshold) using
   Stage 0/1's first-error-step labels as ground truth on a held-out split.
2. Implement mixed-precision decoding: default W4/KV4, dequantize to
   BF16/FP8 for the forward pass at detected pivot steps only.
3. Baselines: uniform W4A4; uniform W8A8; random-step promotion at matched
   promotion rate (controls for "promotion helps regardless of where").

**Metric:** accuracy-recovered vs. %-of-steps-promoted (Pareto curve),
plus measured latency/throughput overhead vs. BF16 and vs. uniform W4A4 in
a real serving stack (vLLM or SGLang), not just simulated FLOPs.

**Success criterion:** better accuracy-vs-average-bit-width Pareto than
uniform W8A8, while promoting a small minority of steps (target: <15%).

## Stage 4 — Joint weight/KV-cache precision co-design

Builds on Stage 3. Long CoT is KV-cache-bound more than weight-bound at
scale, so treat the two precision budgets jointly rather than matching them
by convention (both W4/KV4).

- Sweep weight-bit-width × KV-bit-width jointly under a fixed
  memory/bandwidth budget; find the Pareto-optimal allocation for reasoning
  workloads specifically.
- Extend Stage 3's promotion mechanism to KV-cache entries: promote only
  the KV state written *during* a pivot step, not the whole sequence.
- Requires real serving-stack integration (vLLM/SGLang per-layer KV
  precision control) to get real throughput/memory numbers.

## Execution order

Stage 0 is a hard prerequisite. Stage 1 should run first after that — it's
cheap (inference-only) and determines where Stages 2–4 should focus. Stage
2 and Stage 3 can run in parallel once Stage 1 is done (they don't depend
on each other). Stage 4 depends on Stage 3's promotion mechanism existing.

## Compute notes

Stages 0–2 are feasible on a single multi-GPU node (inference + offline
calibration only). Stage 3 needs a serving stack with custom mixed-precision
kernels — the main engineering risk in this plan. Stage 4 is the most
systems-heavy and should only start once Stage 3 shows a real Pareto win at
the simulated/offline level. Start all experiments at 4B–8B scale for
iteration speed; validate the final result at 32B+ (Qwen3-32B or
DeepSeek-R1-Distill-Llama-70B) before claiming the result generalizes.
