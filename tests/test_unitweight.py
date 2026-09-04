import pytest
from unitweight import weighted_units


def test_reference_size_slice_scores_same_as_unweighted():
    # A reference-sized slice (400 MiB = 419430400 bytes) should score the same as unweighted
    tokens = 100
    layer_start = 0
    layer_end = 10
    slice_bytes = 419430400  # ref_bytes default
    expected = tokens * (layer_end - layer_start)  # = 1000
    assert weighted_units(tokens, layer_start, layer_end, slice_bytes) == expected


def test_10x_larger_slice_scores_10x():
    # A slice 10x larger than reference should score 10x
    tokens = 100
    layer_start = 0
    layer_end = 10
    slice_bytes = 419430400 * 10  # 10x reference
    expected = tokens * (layer_end - layer_start) * 10  # = 10000
    assert weighted_units(tokens, layer_start, layer_end, slice_bytes) == expected


def test_10x_smaller_slice_scores_tenth_rounded():
    # A slice 10x smaller than reference should score 1/10th, rounded
    tokens = 100
    layer_start = 0
    layer_end = 10
    slice_bytes = 41943040  # 1/10th of reference
    expected = tokens * (layer_end - layer_start) // 10  # = 1000, then divided by 10 = 100
    assert weighted_units(tokens, layer_start, layer_end, slice_bytes) == expected


def test_zero_negative_slice_bytes_fallback_to_unweighted():
    # Zero or negative slice_bytes should fall back to unweighted calculation
    tokens = 100
    layer_start = 0
    layer_end = 10
    
    # Test with zero slice_bytes
    assert weighted_units(tokens, layer_start, layer_end, 0) == tokens * (layer_end - layer_start)
    
    # Test with negative slice_bytes
    assert weighted_units(tokens, layer_start, layer_end, -100) == tokens * (layer_end - layer_start)


def test_inverted_layer_range():
    # layer_end <= layer_start should return 0
    tokens = 100
    layer_start = 10
    layer_end = 5
    assert weighted_units(tokens, layer_start, layer_end, 419430400) == 0


def test_zero_tokens():
    # Zero tokens should return 0
    layer_start = 0
    layer_end = 10
    slice_bytes = 419430400
    assert weighted_units(0, layer_start, layer_end, slice_bytes) == 0


def test_non_int_inputs():
    # Non-int inputs should return 0
    assert weighted_units(100.0, 0, 10, 419430400) == 0
    assert weighted_units(100, 0.0, 10, 419430400) == 0
    assert weighted_units(100, 0, 10.0, 419430400) == 0
    assert weighted_units(100, 0, 10, 419430400.0) == 0
    assert weighted_units("100", 0, 10, 419430400) == 0


def test_ref_bytes_zero_uses_default():
    # ref_bytes=0 should use the default value instead of dividing by zero
    tokens = 100
    layer_start = 0
    layer_end = 10
    slice_bytes = 419430400  # same as default
    expected = tokens * (layer_end - layer_start)  # = 1000
    assert weighted_units(tokens, layer_start, layer_end, slice_bytes, ref_bytes=0) == expected


def test_worked_case_with_exact_expected_integer():
    # A worked case with exact expected integer to pin arithmetic
    tokens = 7
    layer_start = 3
    layer_end = 8  # 5 layers
    slice_bytes = 83886080  # 1/5th of reference (419430400 / 5)
    # Expected: 7 × 5 × (83886080 / 419430400) = 7 × 5 × 0.2 = 7
    assert weighted_units(tokens, layer_start, layer_end, slice_bytes) == 7