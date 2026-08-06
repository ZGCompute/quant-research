import math

from pytest import approx

from quant_research.cascade import (
    CascadeResult,
    StepDistribution,
    divergence_position_distribution,
    error_propagation_rate,
    first_error_step,
    score_cascade,
)


def make_step(argmax_token: int, logprobs: dict[int, float]) -> StepDistribution:
    return StepDistribution(argmax_token=argmax_token, token_logprobs=logprobs)


def test_no_divergence():
    reference = [1, 2, 3]
    candidate = [
        make_step(1, {1: -0.1, 2: -3.0}),
        make_step(2, {2: -0.1, 3: -3.0}),
        make_step(3, {3: -0.1, 1: -3.0}),
    ]
    assert first_error_step(reference, candidate) is None

    result = score_cascade(reference, candidate, "42", "42")
    assert result.diverged is False
    assert result.propagated_to_final_answer is False
    assert result.first_error_step is None


def test_divergence_without_answer_change():
    reference = [1, 2, 3]
    candidate = [
        make_step(1, {1: -0.1, 2: -3.0}),
        make_step(9, {9: -0.2, 2: -0.3}),  # diverges here
        make_step(3, {3: -0.1, 1: -3.0}),
    ]
    result = score_cascade(reference, candidate, "42", "42")
    assert result.diverged is True
    assert result.first_error_step == 1
    assert result.propagated_to_final_answer is False
    assert result.divergence_margin == approx(0.1)


def test_divergence_with_answer_change():
    reference = [1, 2, 3]
    candidate = [
        make_step(1, {1: -0.1, 2: -3.0}),
        make_step(9, {9: -0.2, 2: -0.3}),
        make_step(3, {3: -0.1, 1: -3.0}),
    ]
    result = score_cascade(reference, candidate, "42", "7")
    assert result.diverged is True
    assert result.propagated_to_final_answer is True


def test_error_propagation_rate_aggregates_only_diverged_traces():
    results = [
        CascadeResult(first_error_step=None, diverged=False, propagated_to_final_answer=False),
        CascadeResult(first_error_step=2, diverged=True, propagated_to_final_answer=True),
        CascadeResult(first_error_step=5, diverged=True, propagated_to_final_answer=False),
    ]
    assert error_propagation_rate(results) == approx(0.5)


def test_error_propagation_rate_empty_is_zero():
    assert error_propagation_rate([]) == 0.0
    non_diverged = [
        CascadeResult(first_error_step=None, diverged=False, propagated_to_final_answer=False)
    ]
    assert error_propagation_rate(non_diverged) == 0.0


def test_divergence_position_distribution_normalizes_by_trace_length():
    results = [
        CascadeResult(first_error_step=2, diverged=True, propagated_to_final_answer=False),
        CascadeResult(first_error_step=None, diverged=False, propagated_to_final_answer=False),
        CascadeResult(first_error_step=8, diverged=True, propagated_to_final_answer=True),
    ]
    lengths = [10, 10, 10]
    positions = divergence_position_distribution(results, lengths)
    assert positions == [approx(0.2), approx(0.8)]


def test_entropy_uniform_distribution_is_log_n():
    step = make_step(1, {1: math.log(0.5), 2: math.log(0.5)})
    assert step.entropy() == approx(math.log(2))


def test_entropy_certain_distribution_is_zero():
    step = make_step(1, {1: 0.0, 2: -50.0})
    assert step.entropy() == approx(0.0, abs=1e-6)


def test_top2_margin_single_candidate_is_infinite():
    step = make_step(1, {1: -0.1})
    assert step.top2_margin() == math.inf
