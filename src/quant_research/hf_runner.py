"""HF Transformers runner for Stage 0 cascade evaluation against a real model.

cascade.py is intentionally decoupled from any model runtime — this module is
the "caller" referenced in its docstring, responsible for producing
StepDistribution objects from an actual HF Transformers model. It is the only
module in the package that depends on torch/transformers/bitsandbytes (the
`hf` extras group), so importing quant_research.cascade never requires them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from quant_research.cascade import StepDistribution

if TYPE_CHECKING:
    import torch

QuantMode = Literal["bf16", "nf4", "int8"]

_BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
_LAST_INT_RE = re.compile(r"-?\d+")


def extract_boxed_answer(text: str) -> str | None:
    """Pull the contents of the last \\boxed{...} in text (standard MATH/AIME
    answer format). Falls back to the last integer in the text since some
    traces state the answer in prose without a boxed final line."""
    matches = _BOXED_RE.findall(text)
    if matches:
        return matches[-1].strip()
    ints = _LAST_INT_RE.findall(text)
    return ints[-1] if ints else None


@dataclass
class LoadedModel:
    model: object
    tokenizer: object
    quant_mode: QuantMode


def load_model(model_id: str, quant_mode: QuantMode = "bf16") -> LoadedModel:
    """Load model_id under the given quantization mode. Requires the `hf`
    extras. nf4/int8 use bitsandbytes — a calibration-free PTQ stand-in for
    the eventual MR-GPTQ/NVFP4 conditions, good enough for Stage 0's harness
    smoke test on Ampere-class Colab GPUs which lack native FP4 tensor cores."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if quant_mode == "bf16":
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map="auto"
        )
    else:
        from transformers import BitsAndBytesConfig

        bnb_config = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            if quant_mode == "nf4"
            else BitsAndBytesConfig(load_in_8bit=True)
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb_config, device_map="auto"
        )

    model.eval()
    return LoadedModel(model=model, tokenizer=tokenizer, quant_mode=quant_mode)


def generate_reference(
    loaded: LoadedModel, prompt: str, max_new_tokens: int = 1024
) -> tuple[list[int], str]:
    """Free-running greedy generation from the reference model. Returns the
    generated response token ids (prompt stripped) and decoded text."""
    import torch

    inputs = loaded.tokenizer(prompt, return_tensors="pt").to(loaded.model.device)
    with torch.no_grad():
        out = loaded.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    response_ids = out[0][inputs["input_ids"].shape[1] :].tolist()
    text = loaded.tokenizer.decode(response_ids, skip_special_tokens=True)
    return response_ids, text


def logits_to_step_distribution(step_logits: torch.Tensor, topk: int = 5) -> StepDistribution:
    """Convert one position's logits into a StepDistribution. Split out from
    teacher_forced_step_distributions so it's unit-testable with a hand-built
    tensor instead of a loaded model."""
    import torch

    log_probs = torch.log_softmax(step_logits, dim=-1)
    k = min(topk, step_logits.shape[-1])
    top_logprobs, top_ids = torch.topk(log_probs, k)
    token_logprobs = {tid.item(): lp.item() for tid, lp in zip(top_ids, top_logprobs)}
    return StepDistribution(argmax_token=top_ids[0].item(), token_logprobs=token_logprobs)


def teacher_forced_step_distributions(
    loaded: LoadedModel, prompt: str, response_token_ids: list[int], topk: int = 5
) -> list[StepDistribution]:
    """Feed `prompt + response_token_ids` through `loaded.model` in a single
    forward pass and extract this model's own top-k next-token distribution
    at each response position — the teacher-forced replay described in
    EXPERIMENTS.md Stage 0: feed the candidate model the reference prefix and
    compare its next-token distribution to the reference's at that position.

    Also useful as a sanity check when `loaded` IS the reference model: its
    own teacher-forced trace should diverge ~nowhere from the greedy trace it
    produced (mismatches would indicate a generation/scoring-path bug, e.g. a
    tokenizer or dtype mismatch, not a quantization effect)."""
    import torch

    prompt_ids = loaded.tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
    full_ids = prompt_ids + response_token_ids
    input_tensor = torch.tensor([full_ids], device=loaded.model.device)

    with torch.no_grad():
        logits = loaded.model(input_tensor).logits[0]

    prompt_len = len(prompt_ids)
    return [
        logits_to_step_distribution(logits[prompt_len - 1 + i], topk=topk)
        for i in range(len(response_token_ids))
    ]
