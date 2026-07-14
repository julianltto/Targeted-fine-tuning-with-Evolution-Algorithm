"""Mandatory parser cases (CLAUDE.md): comma numbers, $, decimals, negatives,
trailing chatter after the answer, and no-number outputs (wrong, not crash)."""

from evomath.harness.parse import is_correct, parse_gold, parse_pred


GOLD_42 = "Some reasoning here.\n#### 42"


def test_comma_large_numbers():
    assert parse_pred("So the total is 1,234,567 dollars") == 1234567.0
    assert parse_gold("blah\n#### 1,234,567") == 1234567.0
    assert is_correct("The answer is 1,234,567.", "x\n#### 1,234,567")


def test_dollar_sign():
    assert parse_gold("reasoning\n#### $18") == 18.0
    assert is_correct("She has $18 left. The answer is 18.", "r\n#### $18")


def test_decimals():
    assert parse_pred("The answer is 3.5.") == 3.5
    assert is_correct("half of 7 is 3.5. The answer is 3.5.", "r\n#### 3.5")
    # tolerance 1e-4
    assert is_correct("The answer is 3.501.", "r\n#### 3.5") is False   # off by 1e-3
    assert is_correct("The answer is 3.50001.", "r\n#### 3.5")          # off by 1e-5


def test_negative():
    assert parse_pred("The answer is -8.") == -8.0
    assert is_correct("net change is -8. The answer is -8.", "r\n#### -8")


def test_trailing_chatter_takes_last_number():
    out = "The answer is 42. I hope that helps! Let me know if you need 1 more thing"
    # last numeric string wins per spec — chatter containing digits flips the answer
    assert parse_pred(out) == 1.0
    assert not is_correct(out, GOLD_42)
    # without digit chatter the answer survives
    assert is_correct("The answer is 42. Hope that helps!", GOLD_42)


def test_no_number_is_wrong_not_crash():
    assert parse_pred("I cannot solve this problem.") is None
    assert is_correct("I cannot solve this problem.", GOLD_42) is False


def test_gold_requires_number():
    import pytest
    with pytest.raises(ValueError):
        parse_gold("#### no digits at all")
