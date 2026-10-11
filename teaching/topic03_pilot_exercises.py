"""Mechanism-first Topic 03 starter: two CPU-only student edits, no GPU import."""
import math
from typing import Callable, Sequence, TypedDict


class QuantizedWeights(TypedDict):
    """Reconstructed [out,in] weights and row-major group scales, not packed bytes."""
    restored: list[list[float]]
    scales: list[float]


class QuantizationComparison(TypedDict):
    """Errors and ideal payload/FP16-scale byte counts for one population."""
    weight_mse: float
    signed_output_error: list[list[float]]
    output_mse: float
    ideal_code_bytes: int
    fp16_scale_bytes: int


class TimingSummary(TypedDict):
    """Median and observed extremes, all in milliseconds, not confidence bounds."""
    median_ms: float
    minimum_ms: float
    maximum_ms: float

WEIGHTS = [[.49, .40, .60, 7.], [-.31, -.60, .92, -1.40]]
INPUTS = [[1., 1., 0., 0.], [0., 1., 0., 0.]]
HELD_OUT = [[10., 0., 0., 1.]]


def quantize_scalar(value: float, scale: float, zero_point: int = 0) -> tuple[int, float]:
    """Quantize and reconstruct one scalar with the supplied signed INT4 rule.

    Parameters
    ----------
    value : float
        Finite original weight in weight units.
    scale : float
        Positive finite step in the same units.
    zero_point : int, default 0
        Signed INT4 code offset, in [-8,7]; bool is rejected.

    Returns
    -------
    tuple[int, float]
        Clipped nearest-even integer code and reconstructed weight.
        Absmax/7 leaves code -8 unused; this representation is not NF4.

    Raises
    ------
    ValueError
        If value/scale is invalid or zero_point is not a signed INT4 integer.
    """
    if not math.isfinite(value) or not math.isfinite(scale) or scale <= 0:
        raise ValueError('finite value and positive finite scale required')
    if type(zero_point) is not int or not -8 <= zero_point <= 7:
        raise ValueError('signed INT4 zero point required')
    code = max(-8, min(7, round(value / scale) + zero_point))
    return code, scale * (code - zero_point)


def derive_scale(values: Sequence[float]) -> float:
    """Choose a symmetric range-fitting step for one nonempty group.

    Parameters
    ----------
    values : Sequence[float]
        Finite group weights; use seven positive codes to cover their range.

    Returns
    -------
    float
        Positive step in weight units; use 1.0 for an all-zero group.

    Raises
    ------
    ValueError
        If the group is empty or contains a nonfinite value.
    """
    if not values or any(not math.isfinite(v) for v in values):
        raise ValueError('nonempty finite values required')
    # TODO 1: derive the scale, including the all-zero case.
    raise NotImplementedError('Derive a scale from the weights')


def quantize_groups(
    weights: Sequence[Sequence[float]], group_size: int | None = None,
) -> QuantizedWeights:
    """Reconstruct a matrix with global or row-local symmetric INT4 scales.

    Parameters
    ----------
    weights : Sequence[Sequence[float]]
        Nonempty finite rectangular W[out,in]; do not mutate it.
    group_size : int or None, default None
        None uses one global scale. A positive integer uses contiguous
        groups inside each output row, including its shorter final group.

    Returns
    -------
    QuantizedWeights
        restored retains W[out,in]. scales is a row-major flat list:
        one global value or out * ceil(in/group_size) group values.
        Reconstructed Python floats are not physically packed INT4 storage.

    Raises
    ------
    ValueError
        If weights are empty/ragged/nonfinite, or the group size is invalid.

    Notes
    -----
    Use derive_scale and quantize_scalar. In the 2x4 toy, group4 is per-output
    channel and group2 uses two groups per row; groups never cross rows.
    """
    if not weights or not weights[0] or any(len(row) != len(weights[0]) for row in weights):
        raise ValueError('nonempty rectangular [out,in] matrix required')
    if any(not math.isfinite(v) for row in weights for v in row):
        raise ValueError('finite weights required')
    if group_size is not None and (type(group_size) is not int or group_size < 1):
        raise ValueError('positive integer group size or None required')
    # TODO 2: partition, derive each scale, restore every weight.
    raise NotImplementedError('Compare a global scale with row-local groups')


def outputs(weights: Sequence[Sequence[float]], inputs: Sequence[Sequence[float]]) -> list[list[float]]:
    """Apply a bias-free toy linear layer.

    Parameters
    ----------
    weights : Sequence[Sequence[float]]
        W[out,in]; caller supplies matching rectangular feature dimensions.
    inputs : Sequence[Sequence[float]]
        X[samples,in]; features, not token IDs.

    Returns
    -------
    list[list[float]]
        Y[samples,out] = X @ W.T; empty inputs produce an empty list.
    """
    return [[sum(w*x for w, x in zip(row, vector)) for row in weights] for vector in inputs]


def compare(
    weights: Sequence[Sequence[float]], quantized: QuantizedWeights,
    inputs: Sequence[Sequence[float]],
) -> QuantizationComparison:
    """Compare weight error, input-dependent output error and ideal storage.

    Parameters
    ----------
    weights : Sequence[Sequence[float]]
        Original nonempty W[out,in] in weight units.
    quantized : QuantizedWeights
        Reconstructed weights of the same shape and their exact scales.
    inputs : Sequence[Sequence[float]]
        Nonempty X[samples,in] with matching feature dimension.

    Returns
    -------
    QuantizationComparison
        weight_mse averages all out*in squared weight errors.
        signed_output_error has shape [samples,out], approximation-original.
        output_mse averages all samples*out squared output errors.
        ideal_code_bytes packs two codes/byte (rounded up); fp16_scale_bytes
        counts two bytes/scale. Arithmetic does not round scales to FP16.

    Notes
    -----
    Scale rounding, alignment, headers and Python-object sizes are omitted.
    """
    restored = quantized['restored']
    error = [[a-b for a, b in zip(row, original)] for row, original in zip(restored, weights)]
    signed = outputs(error, inputs)
    count = sum(map(len, weights))
    return dict(weight_mse=sum(v*v for row in error for v in row)/count,
                signed_output_error=signed,
                output_mse=sum(v*v for row in signed for v in row)/(len(inputs)*len(weights)),
                ideal_code_bytes=(count+1)//2, fp16_scale_bytes=2*len(quantized['scales']))


def test_scale(function: Callable[[Sequence[float]], float]) -> None:
    """Check range fitting and invalid-group behavior.

    Parameters
    ----------
    function : Callable[[Sequence[float]], float]
        Student derive_scale implementation.

    Returns
    -------
    None
        Silent success; an incorrect result raises AssertionError.
    """
    for values, expected in [([.49,.4,.6,7],1.), ([-1.4,.9],.2),
                             ([0.,0.],1.), ([-7.],1.), ([.49,.4],.07)]:
        assert math.isclose(function(values), expected, abs_tol=1e-12), values
    for values in [[], [float('nan')], [float('inf')]]:
        try:
            function(values)
        except ValueError:
            continue
        raise AssertionError('Invalid range must fail')


def test_groups(function: Callable[[Sequence[Sequence[float]], int | None], QuantizedWeights]) -> None:
    """Check every restored weight, row-local tail and input immutability.

    Parameters
    ----------
    function : Callable
        Student quantize_groups implementation with the documented contract.

    Returns
    -------
    None
        Silent success; incorrect scales, weights or rejection raise AssertionError.
    """
    import copy
    original = copy.deepcopy(WEIGHTS)
    whole = function(WEIGHTS, None)
    assert whole['restored'] == [[0.,0.,1.,7.], [0.,-1.,1.,-1.]]
    expected_scales = {4:[1.,.2], 2:[.07,1.,.6/7,.2], 1:[abs(w)/7 for row in WEIGHTS for w in row]}
    for size, expected in expected_scales.items():
        result = function(WEIGHTS, size)
        assert len(result['scales']) == len(expected)
        assert all(math.isclose(a,b,abs_tol=1e-12) for a,b in zip(result['scales'],expected))
    # Check EVERY reconstructed weight, independently of the returned scales.
    # Different row ranges expose global restoration disguised as row-local.
    for matrix in [WEIGHTS, [[.1,.2,7.],[-.4,-.5,-.6]], [[0.,0.,0.],[1.,-2.,3.]]]:
        before = copy.deepcopy(matrix)
        for size in [None, 1, 2, len(matrix[0]), len(matrix[0])+1]:
            result = function(matrix, size)
            assert len(result['restored']) == len(matrix)
            expected_count = 1 if size is None else len(matrix)*math.ceil(len(matrix[0])/size)
            assert len(result['scales']) == expected_count, 'Each row owns its tail scale'
            global_max = max(abs(v) for row in matrix for v in row)
            scale_index = 0
            for row, restored in zip(matrix, result['restored']):
                assert len(restored) == len(row)
                width = size or len(row)
                for start in range(0, len(row), width):
                    values = row[start:start+width]
                    maximum = global_max if size is None else max(abs(v) for v in values)
                    scale = maximum/7 if maximum else 1.
                    actual_scale = result['scales'][0 if size is None else scale_index]
                    assert math.isclose(actual_scale, scale, rel_tol=1e-12, abs_tol=1e-12)
                    for offset, value in enumerate(values):
                        expected = max(-8, min(7, round(value/scale)))*scale
                        assert math.isclose(restored[start+offset], expected, rel_tol=1e-12, abs_tol=1e-12), 'Wrong group reconstruction'
                    scale_index += 1
            assert matrix == before, 'Do not mutate any input matrix'
    group = function(WEIGHTS, 2)
    assert math.isclose(group['restored'][0][1], .42, abs_tol=1e-12)
    # Output reduction is checked from all restored entries, not a magic loss.
    probe = [[1.,2.,3.,4.], [-1.,0.,2.,-3.]]
    for result in [whole, group]:
        expected = [[sum((q-w)*x for q,w,x in zip(qrow,wrow,vector))
                     for qrow,wrow in zip(result['restored'], WEIGHTS)] for vector in probe]
        measured = compare(WEIGHTS, result, probe)
        assert measured['signed_output_error'] == expected
        assert math.isclose(measured['output_mse'], sum(v*v for row in expected for v in row)/4)
    tail = function([[.1,.2,7.],[.4,.5,.6]], 2)
    assert len(tail['scales']) == 4, 'Tail group cannot cross rows'
    assert [len(row) for row in tail['restored']] == [3,3]
    assert function([[0.,0.]], 2) == {'restored':[[0.,0.]], 'scales':[1.]}
    assert WEIGHTS == original, 'Do not mutate the source matrix'
    for weights,size in [([],2),([[]],2),([[1],[1,2]],2),([[1]],0),([[1]],True),([[float('nan')]],2)]:
        try:
            function(weights,size)
        except ValueError:
            continue
        raise AssertionError('Invalid matrix/group must fail')


def summarize_seconds(values: Sequence[float]) -> TimingSummary:
    """Summarize raw durations without altering their seconds values.

    Parameters
    ----------
    values : Sequence[float]
        Nonempty positive finite durations in seconds.

    Returns
    -------
    TimingSummary
        Median, minimum and maximum in milliseconds; no tail certainty.

    Raises
    ------
    ValueError
        If values are empty, nonnumeric, nonfinite or nonpositive.
    """
    if not values or any(type(v) not in (int,float) or not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError('nonempty positive finite timings required')
    ordered = sorted(values)
    n = len(ordered)
    median = (ordered[(n-1)//2] + ordered[n//2]) / 2
    return dict(median_ms=1000*median, minimum_ms=1000*ordered[0], maximum_ms=1000*ordered[-1])


def test_summary(function: Callable[[Sequence[float]], TimingSummary]) -> None:
    """Check millisecond conversion, median and validation.

    Parameters
    ----------
    function : Callable
        Provided timing summarizer.

    Returns
    -------
    None
        Silent success; incorrect statistics raise AssertionError.
    """
    samples = [.004,.001,.002,.003]
    assert function(samples) == {'median_ms':2.5, 'minimum_ms':1., 'maximum_ms':4.}
    assert samples == [.004,.001,.002,.003], 'Do not mutate raw evidence'
    assert function([.001,.003,.002])['median_ms'] == 2.
    for invalid in [[],[0.],[-1.],[float('nan')],[float('inf')],[True]]:
        try:
            function(invalid)
        except ValueError:
            continue
        raise AssertionError('Reject invalid timings')


def check_lengths(lengths: Sequence[int]) -> None:
    """Keep the baseline and single prescribed length intervention.

    Parameters
    ----------
    lengths : Sequence[int]
        Exactly [24,96] selected output IDs, not visible text characters.

    Returns
    -------
    None
        Silent success; a changed baseline/intervention raises AssertionError.
    """
    assert type(lengths) in (list,tuple) and list(lengths) == [24,96], (
        'Keep baseline 24; change only the second output cap to 96.')
