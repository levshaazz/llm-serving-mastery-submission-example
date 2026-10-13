"""Provided CPU models for Topic05; no inference, timing claims or student answers."""
import math
from typing import Callable, Sequence, TypedDict


class KVAccount(TypedDict):
    """Independent request payload/capacity counts; excludes shared blocks and metadata."""
    blocks: int
    live_positions: int
    capacity_positions: int
    unused_positions: int
    payload_bytes: int
    capacity_bytes: int


class Request(TypedDict):
    """Invented request: integer arrival tick and positive decode iterations, not seconds."""
    id: str
    arrival: int
    decode_steps: int


class Tick(TypedDict):
    """Work during [tick,tick+1); completion/release occurs at tick+1."""
    tick: int
    active: list[str]
    waiting: list[str]
    finished: list[str]


class Schedule(TypedDict):
    """Decode-only deterministic toy: trace, completion ticks and all-attempt residence."""
    mode: str
    ticks: list[Tick]
    completion: dict[str, int]
    residence_ticks: dict[str, int]
    makespan_ticks: int
    idle_slot_ticks: int


REQUESTS: list[Request] = [
    {'id':'A','arrival':0,'decode_steps':5},
    {'id':'B','arrival':0,'decode_steps':1},
    {'id':'C','arrival':0,'decode_steps':1},
    {'id':'D','arrival':2,'decode_steps':2},
]
QWEN = {'layers':24, 'kv_heads':2, 'head_width':64, 'element_bytes':2}


def require_count(value: int, name: str, minimum: int = 0) -> None:
    """Validate one integer without accepting booleans.

    Parameters
    ----------
    value : int
        Count, never a probability or float.
    name : str
        Diagnostic field name.
    minimum : int, default 0
        Inclusive lower bound.

    Returns
    -------
    None
        Raises ValueError for a noninteger or a count below the bound.
    """
    if type(value) is not int or value < minimum:
        raise ValueError(name + ' must be an integer >= ' + str(minimum))


def kv_bytes_per_position(layers: int, kv_heads: int, head_width: int,
                          element_bytes: int) -> int:
    """Compute keys AND values across all layers for one retained position.

    Parameters
    ----------
    layers : int
        Positive number of full-attention layers.
    kv_heads : int
        Positive KV head count, not query heads. No tensor parallelism.
    head_width : int
        Positive scalar elements per KV head.
    element_bytes : int
        Positive bytes per scalar; 2 for this FP16 toy.

    Returns
    -------
    int
        Payload bytes per position; excludes scales, block tables and alignment.

    Raises
    ------
    ValueError
        Invalid dimensions; boolean integers are rejected.
    """
    for name,value in [('layers',layers),('kv_heads',kv_heads),
                       ('head_width',head_width),('element_bytes',element_bytes)]:
        require_count(value,name,1)
    return 2*layers*kv_heads*head_width*element_bytes


def check_account(function: Callable[[Sequence[int], int, int], KVAccount]) -> int:
    """Check the student's independent-per-request block accounting.

    Parameters
    ----------
    function : Callable
        account_kv(lengths, block_size, bytes_per_position). Zero-length requests
        have no block. Sum the independently rounded capacities; do not mutate inputs.

    Returns
    -------
    int
        Number of fixture checks; success is arithmetic, not GPU evidence.
    """
    cases=[([],16,12,0,0),([0],16,12,0,0),([16],16,12,1,16),
           ([17],16,12,2,17),([17,31,33],16,12288,7,81),
           ([1,1],16,12,2,2),([15,16,17],8,12,7,48)]
    for lengths,size,bpp,blocks,live in cases:
        before=list(lengths)
        actual=function(lengths,size,bpp)
        assert actual==dict(blocks=blocks,live_positions=live,
            capacity_positions=blocks*size,unused_positions=blocks*size-live,
            payload_bytes=live*bpp,capacity_bytes=blocks*size*bpp),(lengths,actual)
        assert lengths==before,'Do not mutate request lengths'
    invalid=[([-1],16,12),([True],16,12),([1.5],16,12),([1],0,12),
             ([1],True,12),([1],16,0),([1],16,True)]
    for args in invalid:
        try: function(*args)
        except ValueError: continue
        raise AssertionError('Invalid count accepted: '+repr(args))
    return len(cases)+len(invalid)


def check_admission(function: Callable[[Sequence[Request], int, int], list[str]]) -> int:
    """Check arrival-aware FIFO selection from an already ordered pending queue.

    Parameters
    ----------
    function : Callable
        admit_fifo(pending,tick,free_slots); returns IDs of arrived requests only,
        preserving queue order. This does not model prefill or memory admission.

    Returns
    -------
    int
        Fixture count. Failing IDs/order/count or invalid inputs raise an error.
    """
    cases=[(REQUESTS,0,2,['A','B']),(REQUESTS[2:],1,2,['C']),
           (REQUESTS[2:],2,2,['C','D']),(REQUESTS,0,0,[]),([],0,2,[])]
    for pending,tick,slots,want in cases:
        before=[dict(r) for r in pending]
        assert function(pending,tick,slots)==want
        assert pending==before,'Selection must not mutate the queue'
    for tick,slots in [(-1,1),(0,-1),(True,1),(0,True),(0,1.5)]:
        try: function(REQUESTS,tick,slots)
        except ValueError: continue
        raise AssertionError('Invalid tick/slots accepted')
    return len(cases)+5


def simulate(admit: Callable[[Sequence[Request], int, int], list[str]],
             requests: Sequence[Request], slots: int = 2,
             mode: str = 'continuous') -> Schedule:
    """Compare iteration-refill with one static cohort at a time.

    Parameters
    ----------
    admit : Callable
        Student arrival-aware FIFO selector. Called at every eligible boundary.
    requests : Sequence[Request]
        Unique nonempty IDs in FIFO order, nonnegative arrival ticks and positive
        decode work. Known lengths simulate stop events, not a real length oracle.
    slots : int, default 2
        Positive concurrent decode capacity. No prefill, memory limit or preemption.
    mode : str, default 'continuous'
        continuous refills each tick; static refills only when its cohort empties.

    Returns
    -------
    Schedule
        Complete finite trace; each tick costs one invented unit, NOT GPU seconds.
        Empty input has zero makespan and no completions. Residence includes waiting.

    Raises
    ------
    ValueError
        Invalid requests/mode/slots or selector violation.
    """
    require_count(slots,'slots',1)
    if mode not in ('static','continuous'): raise ValueError('Unknown mode')
    ids=[r['id'] for r in requests]
    if len(set(ids))!=len(ids) or any(not isinstance(i,str) or not i for i in ids):
        raise ValueError('Unique nonempty request IDs required')
    for r in requests:
        require_count(r['arrival'],'arrival')
        require_count(r['decode_steps'],'decode_steps',1)
    pending=[dict(r) for r in requests]
    by_id={r['id']:r for r in requests}
    remaining={r['id']:r['decode_steps'] for r in requests}
    active: list[str]=[]
    ticks: list[Tick]=[]
    completion: dict[str,int]={}
    tick=0
    deadline=sum(r['decode_steps'] for r in requests)+max([0]+[r['arrival'] for r in requests])+1
    while pending or active:
        if tick>deadline: raise ValueError('Selector made no bounded progress')
        if mode=='continuous' or not active:
            selected=admit(pending,tick,slots-len(active))
            eligible=[r['id'] for r in pending if r['arrival']<=tick]
            if len(set(selected))!=len(selected) or len(selected)>slots-len(active) or any(i not in eligible for i in selected):
                raise ValueError('Invalid admission')
            active.extend(selected)
            pending=[r for r in pending if r['id'] not in selected]
        worked=list(active)
        finished=[]
        for i in active:
            remaining[i]-=1
            if remaining[i]==0: completion[i]=tick+1;finished.append(i)
        ticks.append(dict(tick=tick,active=worked,
            waiting=[r['id'] for r in pending if r['arrival']<=tick],finished=finished))
        active=[i for i in active if i not in finished]
        tick+=1
    return dict(mode=mode,ticks=ticks,completion=completion,
        residence_ticks={i:completion[i]-by_id[i]['arrival'] for i in ids},
        makespan_ticks=tick,idle_slot_ticks=sum(slots-len(t['active']) for t in ticks))


def scalar_attention(keys: Sequence[float], values: Sequence[float],
                     query: float = 1.0) -> tuple[list[float], float]:
    """Provided D=1 one-query attention with one GLOBAL normalization.

    Parameters
    ----------
    keys : Sequence[float]
        Nonempty retained scalar keys, all legal for this final causal query.
    values : Sequence[float]
        Matching scalar values; keys and values must preserve pairing.
    query : float, default 1.0
        Finite scalar query. sqrt(D)=1; no positional rotation in this invented toy.

    Returns
    -------
    tuple[list[float], float]
        Stable global softmax weights and weighted value sum in Python float.

    Raises
    ------
    ValueError
        Empty/mismatched/nonfinite inputs or overflow in query-key scores.
    """
    if not keys or len(keys)!=len(values) or any(not math.isfinite(v) for v in [*keys,*values,query]):
        raise ValueError('Finite matching nonempty inputs required')
    scores=[query*k for k in keys]
    if not all(math.isfinite(score) for score in scores):
        raise ValueError('Attention scores overflowed; use a bounded scalar toy')
    peak=max(scores)
    masses=[math.exp(s-peak) for s in scores]
    weights=[m/sum(masses) for m in masses]
    return weights,sum(w*v for w,v in zip(weights,values))
