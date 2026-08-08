import pytest

torch = pytest.importorskip("torch")

from quant_research.hf_runner import extract_boxed_answer, logits_to_step_distribution


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
