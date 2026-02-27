"""Tests for tunebench.metrics."""

import math

import pytest

from tunebench.metrics import loss_to_perplexity


def test_loss_to_perplexity_zero():
    assert loss_to_perplexity(0.0) == 1.0


def test_loss_to_perplexity_positive():
    assert loss_to_perplexity(1.0) == pytest.approx(math.e)
    assert loss_to_perplexity(2.0) == pytest.approx(math.e**2)


def test_loss_to_perplexity_caps_overflow():
    # Very large loss should be capped to avoid overflow
    result = loss_to_perplexity(1000.0)
    assert math.isfinite(result)
    assert result == pytest.approx(math.exp(100.0))


def test_loss_to_perplexity_nan():
    assert math.isnan(loss_to_perplexity(float("nan")))


def test_loss_to_perplexity_none():
    assert math.isnan(loss_to_perplexity(None))
