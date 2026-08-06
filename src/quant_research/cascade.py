"""Stage 0 cascade evaluation harness.

Measures *where* a quantized model's reasoning trace diverges from a
reference (typically BF16) trace and whether that divergence changes the
final answer, rather than only scoring final-answer accuracy.

Design note: this module is intentionally decoupled from any specific model
runtime (HF Transformers, vLLM, SGLang, ...). Callers are responsible for
producing per-step logprob distributions; this module only implements the
metric logic, so it can be unit-tested without a GPU or a model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class StepDistribution:
    """The candidate model's distribution over the vocabulary at one
    decoding step, reduced to what the cascade metrics need.

    token_logprobs maps token id -> logprob and must contain at least the
    top-1 and top-2 candidates (does not need the full vocabulary).
    """

    argmax_token: int
    token_logprobs: dict[int, float]

    def top2_margin(self) -> float:
        """Difference between the top-1 and top-2 logprob. Large margin =
        confident step; small margin = a step worth flagging as a
        candidate pivot for Stage 3's adaptive-precision promotion."""
        if len(self.token_logprobs) < 2:
            return math.inf
        ordered = sorted(self.token_logprobs.values(), reverse=True)
        return ordered[0] - ordered[1]

    def entropy(self) -> float:
        """Entropy in nats over the provided logprob candidates. Only a
        lower bound on true entropy if token_logprobs is truncated to
        top-k, but consistent enough to compare across steps/conditions."""
        probs = [math.exp(lp) for lp in self.token_logprobs.values()]
        total = sum(probs)
        if total <= 0:
            return 0.0
        return -sum((p / total) * math.log(p / total) for p in probs if p > 0)


@dataclass(frozen=True)
class CascadeResult:
    first_error_step: int | None
    diverged: bool
    propagated_to_final_answer: bool
    divergence_margin: float | None = field(default=None)
    divergence_entropy: float | None = field(default=None)


def first_error_step(
    reference_tokens: list[int],
    candidate_steps: list[StepDistribution],
) -> int | None:
    """First index where the candidate's argmax token differs from the
    reference trace's token at the same position. Returns None if the
    candidate trace matches the reference everywhere they overlap."""
    for i, (ref_tok, cand) in enumerate(zip(reference_tokens, candidate_steps)):
        if cand.argmax_token != ref_tok:
            return i
    return None


def score_cascade(
    reference_tokens: list[int],
    candidate_steps: list[StepDistribution],
    reference_answer: str,
    candidate_answer: str,
) -> CascadeResult:
    """Score a single (reference trace, candidate trace) pair.

    `propagated_to_final_answer` is only meaningful when `diverged` is
    True — a candidate can get the wrong final answer without ever
    diverging in argmax token (e.g. correct tokens, wrong extraction), which
    this function reports as diverged=False and is out of scope for the
    cascade metric (it's not a quantization-induced cascade).
    """
    step = first_error_step(reference_tokens, candidate_steps)
    diverged = step is not None
    propagated = diverged and (candidate_answer != reference_answer)

    margin = candidate_steps[step].top2_margin() if diverged else None
    entropy = candidate_steps[step].entropy() if diverged else None

    return CascadeResult(
        first_error_step=step,
        diverged=diverged,
        propagated_to_final_answer=propagated,
        divergence_margin=margin,
        divergence_entropy=entropy,
    )


def error_propagation_rate(results: list[CascadeResult]) -> float:
    """Fraction of diverged traces whose divergence changed the final
    answer. Returns 0.0 if no traces diverged (undefined in the strict
    sense, but 0.0 is the useful default for aggregation)."""
    diverged = [r for r in results if r.diverged]
    if not diverged:
        return 0.0
    propagated = sum(1 for r in diverged if r.propagated_to_final_answer)
    return propagated / len(diverged)


def divergence_position_distribution(
    results: list[CascadeResult], trace_lengths: list[int]
) -> list[float]:
    """Normalized position (0.0 = start of trace, 1.0 = end) of each
    diverged trace's first-error-step, for comparing whether a
    quantization condition tends to fail early or late in reasoning."""
    positions = []
    for r, length in zip(results, trace_lengths):
        if r.diverged and length > 0:
            positions.append(r.first_error_step / length)
    return positions
