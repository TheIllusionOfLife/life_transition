"""Unit tests for the V4 policy evolution analysis script.

Tests cover:
- compute_weight_delta: per-weight deviation statistics
- validate_output_schema: required-key presence check

The analysis script is imported via sys.path manipulation to mirror the
project convention used by other test modules in this directory.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

# Make the scripts directory importable (mirrors other test modules in this dir)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from analyze_v4_policy_evolution import (  # noqa: E402
    _INITIAL_POLICY,
    compute_weight_delta,
    validate_output_schema,
)

# ---------------------------------------------------------------------------
# compute_weight_delta
# ---------------------------------------------------------------------------


def test_compute_weight_delta_identity():
    """All evolved policies equal to initial → all per-weight deltas must be ~0."""
    initial = _INITIAL_POLICY
    # Three entities, each identical to the initial policy
    vectors = [list(initial), list(initial), list(initial)]
    result = compute_weight_delta(vectors, initial)

    assert result["n_entities"] == 3
    assert len(result["weight_mean_delta"]) == len(initial)
    assert len(result["weight_std_delta"]) == len(initial)

    for i, d in enumerate(result["weight_mean_delta"]):
        assert d == pytest.approx(0.0, abs=1e-9), (
            f"weight_mean_delta[{i}] should be 0.0 for identical policies, got {d}"
        )
    for i, s in enumerate(result["weight_std_delta"]):
        assert s == pytest.approx(0.0, abs=1e-9), (
            f"weight_std_delta[{i}] should be 0.0 for identical policies, got {s}"
        )


def test_compute_weight_delta_shift():
    """If all entities have w0=1.0 (vs initial 0.5), delta_w0 must equal 0.5."""
    initial = _INITIAL_POLICY
    # Shift only w0 from 0.5 → 1.0 across three entities
    vectors = [
        [1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    result = compute_weight_delta(vectors, initial)

    assert result["n_entities"] == 3
    assert result["weight_mean_delta"][0] == pytest.approx(0.5, abs=1e-9), (
        f"delta_w0 should be 0.5, got {result['weight_mean_delta'][0]}"
    )
    # All other weights unchanged → deltas should be 0
    for i in range(1, len(initial)):
        assert result["weight_mean_delta"][i] == pytest.approx(0.0, abs=1e-9), (
            f"weight_mean_delta[{i}] should be 0.0, got {result['weight_mean_delta'][i]}"
        )
    # Std across identical entities must be 0 for all weights
    for i, s in enumerate(result["weight_std_delta"]):
        assert s == pytest.approx(0.0, abs=1e-9), f"weight_std_delta[{i}] should be 0.0, got {s}"


def test_compute_weight_delta_single_entity():
    """Single entity input must work; std must be 0.0 (not NaN or an error)."""
    initial = _INITIAL_POLICY
    evolved = [0.8, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]
    result = compute_weight_delta([evolved], initial)

    assert result["n_entities"] == 1
    assert result["weight_mean_delta"][0] == pytest.approx(0.8 - 0.5, abs=1e-9)
    assert result["weight_mean_delta"][1] == pytest.approx(0.3 - 0.5, abs=1e-9)
    assert result["weight_mean_delta"][2] == pytest.approx(0.1 - 0.0, abs=1e-9)

    for i, s in enumerate(result["weight_std_delta"]):
        assert not math.isnan(s), f"weight_std_delta[{i}] must not be NaN for single entity"
        assert s == pytest.approx(0.0, abs=1e-9), (
            f"weight_std_delta[{i}] should be 0.0 for single entity, got {s}"
        )


def test_compute_weight_delta_empty():
    """Empty policy list must return zero deltas and n_entities=0 without error."""
    initial = _INITIAL_POLICY
    result = compute_weight_delta([], initial)

    assert result["n_entities"] == 0
    assert len(result["weight_mean_delta"]) == len(initial)
    assert len(result["weight_std_delta"]) == len(initial)
    for d in result["weight_mean_delta"]:
        assert d == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# validate_output_schema
# ---------------------------------------------------------------------------


def _make_valid_result() -> dict:
    """Return a minimal valid output dict matching the expected schema."""
    return {
        "analysis": "v4_policy_evolution",
        "seeds": "100-199",
        "initial_policy": _INITIAL_POLICY,
        "harshness_results": {
            "rich": {
                "n_entities": 42,
                "n_seeds": 15,
                "weight_mean_delta": [0.0] * 8,
                "weight_std_delta": [0.0] * 8,
                "magnitude_mean": 0.71,
                "magnitude_std": 0.05,
            }
        },
    }


def test_output_schema_valid():
    """A result dict with all required keys must pass schema validation."""
    result = _make_valid_result()
    assert validate_output_schema(result) is True


def test_output_schema_missing_harshness_results():
    """A result dict missing 'harshness_results' must fail schema validation."""
    result = _make_valid_result()
    del result["harshness_results"]
    assert validate_output_schema(result) is False


def test_output_schema_missing_analysis_key():
    """A result dict missing 'analysis' must fail schema validation."""
    result = _make_valid_result()
    del result["analysis"]
    assert validate_output_schema(result) is False


def test_output_schema_missing_initial_policy():
    """A result dict missing 'initial_policy' must fail schema validation."""
    result = _make_valid_result()
    del result["initial_policy"]
    assert validate_output_schema(result) is False


def test_output_schema_missing_seeds():
    """A result dict missing 'seeds' must fail schema validation."""
    result = _make_valid_result()
    del result["seeds"]
    assert validate_output_schema(result) is False


def test_output_schema_extra_keys_allowed():
    """A result dict with extra keys beyond the required set must still pass."""
    result = _make_valid_result()
    result["extra_metadata"] = "some exploratory note"
    assert validate_output_schema(result) is True
