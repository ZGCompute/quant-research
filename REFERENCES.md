# References

Curated from the landscape survey that motivated this project. Links only
(no PDFs committed to the repo) — grouped by relevance to the experiment
stages in EXPERIMENTS.md.

## Reasoning / long-CoT quantization degradation (core to Stages 0–4)

- [Quantization Meets Reasoning: Exploring and Mitigating Degradation of Low-Bit LLMs in Mathematical Reasoning](https://arxiv.org/pdf/2505.11574) — documents that PTQ disproportionately elevates execution/method errors over conceptual ones, with failures emerging at an identifiable early step that cascades. Primary motivation for Stage 0/1.
- [Quantized Reasoning Models Think They Need to Think Longer, but They Do Not](https://arxiv.org/abs/2606.00206) — quantization increases CoT length while reducing accuracy; models sometimes reach the right answer internally but fail to surface it.
- [Quantization Hurts Reasoning? An Empirical Study on Quantized Reasoning Models](https://arxiv.org/html/2504.04823v2) — broad empirical baseline for reasoning-specific degradation.
- [CUSUM-Shaped Inference-Time Monitoring and Targeted Re-Decoding for Quantized Small Language Model Reasoning](https://arxiv.org/html/2607.20129v1) — closest existing work to Stage 3's adaptive-decoding idea; current approach is a monitoring patch, not a trained detector — worth understanding exactly where it falls short.
- [ParoQuant: Pairwise Rotation Quantization for Efficient Reasoning LLM Inference](https://arxiv.org/pdf/2511.10645) — reasoning-targeted rotation quantization, relevant baseline for Stage 2.

## Hardware-native FP4 formats & outlier smoothing (baseline methods for Stage 1/2)

- [Block Rotation is All You Need for MXFP4 Quantization](https://arxiv.org/pdf/2511.04214) — block-wise Hadamard rotation (MR-GPTQ), current best MXFP4/NVFP4 outlier smoothing; primary quantization baseline for this project.
- [SharQ: Bridging Activation Sparsity and FP4 Quantization for LLM Inference](https://arxiv.org/pdf/2606.26587)
- [SOAR: Scale Optimization for Accurate Reconstruction in NVFP4 Quantization](https://arxiv.org/pdf/2605.12245)
- [MixFP4: Enhancing NVFP4 with Adaptive FP4/INT4 Block Representations](https://arxiv.org/pdf/2605.31035)
- [Diagnosing FP4 inference: a layer-wise and block-wise sensitivity analysis of NVFP4 and MXFP4](https://arxiv.org/pdf/2603.08747) — sensitivity-analysis methodology directly reusable for Stage 2's cascade-aware calibration.
- [HiFloat4 Format for Language Model Pre-training on Ascend NPUs](https://arxiv.org/pdf/2604.08826)

## MoE-specific quantization (context / not the primary target, see README rationale)

- [EAQuant: Enhancing Post-Training Quantization for MoE Models via Expert-Aware Optimization](https://arxiv.org/abs/2506.13329)
- [CodeQuant: Unified Clustering and Quantization for Enhanced Outlier Smoothing in Low-Precision Mixture-of-Experts](https://arxiv.org/abs/2604.10496)
- [Value-and-Structure Alignment for Routing-Consistent Quantization of Mixture-of-Experts Models](https://arxiv.org/pdf/2606.05688)
- [Dynamic Expert Quantization for Scalable Mixture-of-Experts Inference](https://arxiv.org/pdf/2511.15015)
- [Beyond Independent Optimization: Compression, MoE Routing, and Quantization Interactions in Multimodal Edge Intelligence](https://arxiv.org/html/2607.20981v1)

## VLM / modality-heterogeneous quantization (context)

- [Breaking Modality Heterogeneity in Low-Bit Quantization for Large Vision-Language Models (SplitQ)](https://arxiv.org/abs/2605.19929)
- [SPEED-Q: Staged Processing with Enhanced Distillation towards Efficient Low-bit On-device VLM Quantization](https://arxiv.org/pdf/2511.08914)
- [VLMQ: Token Saliency-Driven Post-Training Quantization for Vision-language Models](https://arxiv.org/pdf/2508.03351)
- [VEQ: Modality-Adaptive Quantization for MoE Vision-Language Models](https://arxiv.org/pdf/2602.01037)
- [MBQ: Modality-Balanced Quantization for Large Vision-Language Models](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_MBQ_Modality-Balanced_Quantization_for_Large_Vision-Language_Models_CVPR_2025_paper.pdf)

## To review next

Search coverage above was landscape-level (August 2026 web search), not a
systematic lit review. Before Stage 1 starts, do a citation-graph pass on
the reasoning-degradation papers above to check for anything they cite or
are cited by that changes the plan.
