"""Mechanism-first Topic 03 starter: two CPU-only student edits, no GPU import."""
import math

WEIGHTS = [[.49, .40, .60, 7.], [-.30, -.60, .90, -1.40]]
INPUTS = [[1., 1., 0., 0.], [0., 1., 0., 0.]]
HELD_OUT = [[10., 0., 0., 1.]]


def quantize_scalar(value, scale, zero_point=0):
    """Provided signed INT4; nearest-even, clip [-8,7], reconstruct.
    Absmax/7 fits a symmetric range, leaving stored code -8 unused. Not NF4.
    """
    if not math.isfinite(value) or not math.isfinite(scale) or scale <= 0:
        raise ValueError('finite value and positive finite scale required')
    if type(zero_point) is not int or not -8 <= zero_point <= 7:
        raise ValueError('signed INT4 zero point required')
    code = max(-8, min(7, round(value / scale) + zero_point))
    return code, scale * (code - zero_point)


def derive_scale(values):
    """Return max(abs(values))/7, or 1.0 for an all-zero group.
    STUDENT: derive the step from seven positive codes and the group's range.
    """
    if not values or any(not math.isfinite(v) for v in values):
        raise ValueError('nonempty finite values required')
    # TODO 1: derive the scale, including the all-zero case.
    raise NotImplementedError('Derive a scale from the weights')


def quantize_groups(weights, group_size=None):
    """Return {'restored': rectangular matrix, 'scales': flat list}.
    None: ONE scale for the complete matrix. Positive integer: contiguous chunks
    within EACH output row; never cross rows. Keep the shorter final group.
    For our 2x4 matrix: size 4 = per-output-channel; size 2 = two groups per row.
    Use derive_scale and quantize_scalar. Do not mutate weights. This returns
    reconstructed floats, not physically packed codes.
    """
    if not weights or not weights[0] or any(len(row) != len(weights[0]) for row in weights):
        raise ValueError('nonempty rectangular [out,in] matrix required')
    if any(not math.isfinite(v) for row in weights for v in row):
        raise ValueError('finite weights required')
    if group_size is not None and (type(group_size) is not int or group_size < 1):
        raise ValueError('positive integer group size or None required')
    # TODO 2: partition, derive each scale, restore every weight.
    raise NotImplementedError('Compare a global scale with row-local groups')


def outputs(weights, inputs):
    """[samples,in] times transpose([out,in]) -> [samples,out], no bias."""
    return [[sum(w*x for w, x in zip(row, vector)) for row in weights] for vector in inputs]


def compare(weights, quantized, inputs):
    """Provided MSE/accounting helper. Mean over ALL scalar entries.
    Ideal storage assumes FP16 scales; arithmetic uses exact Python scales.
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


def test_scale(function):
    for values, expected in [([.49,.4,.6,7],1.), ([-1.4,.9],.2),
                             ([0.,0.],1.), ([-7.],1.), ([.49,.4],.07)]:
        assert math.isclose(function(values), expected, abs_tol=1e-12), values
    for values in [[], [float('nan')], [float('inf')]]:
        try:
            function(values)
        except ValueError:
            continue
        raise AssertionError('Invalid range must fail')


def test_groups(function):
    import copy
    original = copy.deepcopy(WEIGHTS)
    whole = function(WEIGHTS, None)
    assert whole['restored'] == [[0.,0.,1.,7.], [0.,-1.,1.,-1.]]
    expected_scales = {4:[1.,.2], 2:[.07,1.,.6/7,.2], 1:[abs(w)/7 for row in WEIGHTS for w in row]}
    for size, expected in expected_scales.items():
        result = function(WEIGHTS, size)
        assert len(result['scales']) == len(expected)
        assert all(math.isclose(a,b,abs_tol=1e-12) for a,b in zip(result['scales'],expected))
    group = function(WEIGHTS, 2)
    assert math.isclose(group['restored'][0][1], .42, abs_tol=1e-12)
    assert math.isclose(compare(WEIGHTS, whole, INPUTS)['output_mse'], .280525, abs_tol=1e-12)
    assert math.isclose(compare(WEIGHTS, group, INPUTS)['output_mse'], .0006591836734693878, abs_tol=1e-12)
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


def summarize_seconds(values):
    """PROVIDED: validated median/range in ms; never mutate raw seconds."""
    if not values or any(type(v) not in (int,float) or not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError('nonempty positive finite timings required')
    ordered = sorted(values)
    n = len(ordered)
    median = (ordered[(n-1)//2] + ordered[n//2]) / 2
    return dict(median_ms=1000*median, minimum_ms=1000*ordered[0], maximum_ms=1000*ordered[-1])


def test_summary(function):
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


def check_lengths(lengths):
    assert type(lengths) in (list,tuple) and list(lengths) == [24,96], (
        'Keep baseline 24; change only the second output cap to 96.')
