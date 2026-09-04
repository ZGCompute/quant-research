import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from quant_research.hf_runner import (
    LoadedModel,
    extract_boxed_answer,
    logits_to_step_distribution,
    teacher_forced_step_distributions,
)


def test_extract_boxed_answer_simple():
    assert extract_boxed_answer("some reasoning... the answer is \\boxed{42}.") == "42"


def test_extract_boxed_answer_nested_braces():
    text = "so the answer is \\boxed{\\frac{1}{2}} exactly"
    assert extract_boxed_answer(text) == "\\frac{1}{2}"


def test_extract_boxed_answer_uses_last_occurrence():
    text = "draft: \\boxed{7} ... final: \\boxed{9}"
    assert extract_boxed_answer(text) == "9"


def test_extract_boxed_answer_falls_back_to_last_integer():
    assert extract_boxed_answer("after simplifying, we get 17") == "17"


def test_extract_boxed_answer_returns_none_when_nothing_found():
    assert extract_boxed_answer("no numeric answer here") is None


def test_logits_to_step_distribution_argmax_matches_highest_logit():
    logits = torch.tensor([1.0, 5.0, 2.0, 0.0])
    dist = logits_to_step_distribution(logits, topk=3)
    assert dist.argmax_token == 1
    assert len(dist.token_logprobs) == 3
    assert dist.token_logprobs[1] == max(dist.token_logprobs.values())


def test_logits_to_step_distribution_topk_clamped_to_vocab_size():
    logits = torch.tensor([1.0, 2.0])
    dist = logits_to_step_distribution(logits, topk=5)
    assert len(dist.token_logprobs) == 2


def _tiny_gpt2_loaded_model() -> LoadedModel:
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=64,
        n_positions=64,
        n_embd=16,
        n_layer=2,
        n_head=2,
        attn_implementation="eager",
    )
    model = GPT2LMHeadModel(config)
    model.eval()
    return LoadedModel(model=model, tokenizer=None, quant_mode="bf16")


def test_teacher_forced_step_distributions_chunking_matches_single_shot():
    """The chunked, KV-cached forward pass must produce bit-for-bit the same
    distributions as one whole-sequence forward pass — chunking is a memory
    optimization, not a behavior change (see the function's docstring)."""
    torch.manual_seed(0)
    loaded = _tiny_gpt2_loaded_model()

    prompt_ids = [1, 2, 3, 4, 5]
    response_ids = list(range(6, 6 + 20))  # 20 response tokens
    full_ids = prompt_ids + response_ids

    class _FakeTokenizer:
        def __call__(self, prompt, return_tensors=None):
            class _Out:
                input_ids = torch.tensor([prompt_ids])

            return _Out()

    loaded.tokenizer = _FakeTokenizer()

    # Ground truth: single-shot whole-sequence forward pass, no chunking.
    with torch.no_grad():
        full_logits = loaded.model(torch.tensor([full_ids])).logits[0]
    expected = [
        logits_to_step_distribution(full_logits[len(prompt_ids) - 1 + i])
        for i in range(len(response_ids))
    ]

    # A chunk size smaller than both the prompt and the response forces
    # multiple chunks on both sides of the prompt/response boundary.
    actual = teacher_forced_step_distributions(
        loaded, prompt="unused", response_token_ids=response_ids, chunk_size=7
    )

    assert len(actual) == len(expected)
    for exp, act in zip(expected, actual):
        assert exp.argmax_token == act.argmax_token
        assert exp.token_logprobs.keys() == act.token_logprobs.keys()
        for tok in exp.token_logprobs:
            assert exp.token_logprobs[tok] == pytest.approx(
                act.token_logprobs[tok], abs=1e-4
            )
