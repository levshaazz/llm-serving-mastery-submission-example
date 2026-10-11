"""Provided CPU infrastructure, exercises and checks. No student solutions.

Finite, single-choice, text-only chat is the teaching scope. A full OpenAI client,
tool calls, reconnection and production timeout enforcement are not implemented here.
"""
import json
from typing import Any, Callable, Sequence
from teaching.topic04_sse import SSEDecoder


def budget_inputs(prompt_tokens: int, output_cap: int, context_limit: int) -> None:
    """Validate counts independently of the student's admission calculation.

    Parameters
    ----------
    prompt_tokens : int
        Nonnegative rendered input-ID count, including template markers.
    output_cap : int
        Positive requested maximum selected output IDs.
    context_limit : int
        Positive serving context capacity, measured in positions.

    Returns
    -------
    None
        Invalid integers (including bools) raise ValueError; no admission answer.
    """
    for name, value in (("prompt_tokens", prompt_tokens), ("output_cap", output_cap),
                        ("context_limit", context_limit)):
        if type(value) is not int or value < 0:
            raise ValueError(name + " must be a nonnegative integer")
    if context_limit == 0:
        raise ValueError("context_limit must be positive")
    if output_cap == 0:
        raise ValueError("output_cap must be positive for this vLLM generation request")


def decode_chunks(chunks: Sequence[bytes]) -> list[dict[str, Any] | str]:
    """Incremental UTF-8 + SSE framing; leave lifecycle decisions to the student.

    Arrival clocks are intentionally omitted: resegmentation keeps bytes, not timings.
    Only the finite teaching subset (JSON data or [DONE]) is returned.

    Parameters
    ----------
    chunks : sequence of bytes
        Contiguous wire bytes in transport order; arbitrary segmentation.

    Returns
    -------
    list[dict or str]
        JSON envelopes and literal [DONE], without invented arrival clocks.
    """
    decoder = SSEDecoder()
    records = []
    for raw in chunks:
        for frame in decoder.feed(raw):
            if frame['event'] != 'message':
                raise ValueError('Only finite message events supported')
            data = frame['data']
            records.append("[DONE]" if data == "[DONE]" else json.loads(data))
    if decoder.finish():
        raise ValueError('Unfinished SSE frame at EOF; do not call it complete')
    return records


def check_admission(admit: Callable[[int, int, int], bool]) -> int:
    """Check valid and invalid positional-budget fixtures.

    Parameters
    ----------
    admit : callable
        Student predicate over rendered input, output cap and context limit.

    Returns
    -------
    int
        Passing fixture count; wrong results raise AssertionError.
    """
    cases = [(33, 16, 2048, True), (1800, 512, 2048, False),
             (1800, 128, 2048, True), (2329, 16, 2048, False),
             (2032, 16, 2048, True), (2033, 16, 2048, False)]
    for p, o, limit, expected in cases:
        actual = admit(p, o, limit)
        assert type(actual) is bool and actual is expected, (p, o, limit, actual)
    for args in [(True, 1, 2048), (-1, 1, 2048), (1, 1.5, 2048), (1, 1, 0), (1, 0, 2048)]:
        try:
            admit(*args)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid budget input accepted: " + repr(args))
    return len(cases) + 5


def event(content: str | None = None, finish: str | None = None) -> dict[str, Any]:
    """Construct one invented single-choice lifecycle envelope.

    Parameters
    ----------
    content : str or None
        Optional text delta; None produces an empty delta.
    finish : str or None
        Optional terminal reason, independent of text validity.

    Returns
    -------
    dict
        OpenAI-style choices array; not server evidence.
    """
    delta = {} if content is None else {"content": content}
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


def check_reducer(reduce_stream: Callable[[Sequence[dict[str, Any] | str]], dict[str, Any]]) -> int:
    """Supported envelopes; the exercise owns concatenation and success policy.

    Malformed JSON/UTF-8 and the broader server protocol are the provided parser's
    separate responsibility. Duplicate/out-of-order lifecycle data must fail.

    Parameters
    ----------
    reduce_stream : callable
        Maps ordered envelopes to content, status, done and finish_reason.

    Returns
    -------
    int
        Number of passing lifecycle fixtures; failed checks raise AssertionError.
    """
    prefix = [{"choices": [{"index": 0, "delta": {"role": "assistant"},
                             "finish_reason": None}]}, event("Hello"), event(" world")]
    usage = {"choices": [], "usage": {"prompt_tokens": 33,
              "completion_tokens": 10, "total_tokens": 43}}
    cases = [
        (prefix + [event(finish="stop"), usage, "[DONE]"], "Hello world", "complete"),
        (prefix, "Hello world", "partial"),
        (prefix + [event(finish="stop")], "Hello world", "partial"),
        (prefix + ["[DONE]"], "Hello world", "failed"),
        (prefix + [{"error": {"message": "engine failed"}}, event(" late"), "[DONE]"], "Hello world", "failed"),
        (prefix + [event(finish="length"), "[DONE]"], "Hello world", "complete"),
        (prefix + [event(finish="stop"), event(" late"), "[DONE]"], "Hello world", "failed"),
        (prefix + [event(finish="stop"), "[DONE]", event(" late")], "Hello world", "failed"),
        (prefix + [event(finish="stop"), event(finish="stop"), "[DONE]"], "Hello world", "failed"),
    ]
    for records, content, status in cases:
        result = reduce_stream(records)
        assert result["content"] == content, (records, result)
        assert result["status"] == status, (records, result)
        assert type(result["done"]) is bool and result["done"] is ("[DONE]" in records), (records, result)
        reasons = [c.get("finish_reason") for r in records if isinstance(r, dict)
                   for c in r.get("choices", []) if c.get("finish_reason") is not None]
        assert result["finish_reason"] == (reasons[-1] if reasons else None), (records, result)
    return len(cases)


def byte_partitions(wire: bytes) -> dict[str, list[bytes]]:
    """Segment the same transcript three ways without a network operation.

    Parameters
    ----------
    wire : bytes
        Complete wire transcript, possibly empty.

    Returns
    -------
    dict[str, list[bytes]]
        one_read, one_byte and seventeen_bytes transport partitions.
    """
    return {"one_read": [wire], "one_byte": [bytes([v]) for v in wire],
            "seventeen_bytes": [wire[i:i+17] for i in range(0, len(wire), 17)]}
