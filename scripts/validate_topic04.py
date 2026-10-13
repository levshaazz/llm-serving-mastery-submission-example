"""Bounded UTF-8/SSE decoder + single-choice chat lifecycle, standard library only.

Framing follows WHATWG event streams (CR/LF/CRLF, comments, multiline data).
Chat completion is a separate, deliberately strict course contract, not all SSE.
"""
import codecs
import json
import math
from typing import Any, TypedDict


class SSEEvent(TypedDict):
    """One framed event; data remains uninterpreted text."""
    event: str
    data: str


class ChatResult(TypedDict):
    """Finite lifecycle observation; all times are client-arrival seconds."""
    status: str
    content: str
    finish_reason: str | None
    done: bool
    usage: dict[str, int] | None
    first_event_s: float | None
    first_content_s: float | None
    wall_s: float
    byte_count: int
    event_count: int
    events: list[dict[str, Any]]
    error: str | None
    unfinished_frame: bool
    clock: str


class SSEProtocolError(ValueError):
    pass


class SSEDecoder:
    def __init__(self, max_bytes: int = 262144, max_events: int = 4096) -> None:
        """Initialize a bounded, incremental UTF-8 decoder.

        Parameters
        ----------
        max_bytes : int
            Cumulative byte budget, not the size of one transport chunk.
        max_events : int
            Maximum number of dispatched events.

        Returns
        -------
        None
            Initializes framing state; performs no network operation.
        """
        self.decoder = codecs.getincrementaldecoder('utf-8-sig')('strict')
        self.line = ''; self.data = []; self.event = ''; self.after_cr = False
        self.bytes = 0; self.count = 0; self.max_bytes = max_bytes; self.max_events = max_events
        self.closed = False

    def _line(self) -> SSEEvent | None:
        """Consume one buffered line according to SSE framing.

        Returns
        -------
        SSEEvent or None
            A dispatched event on a blank line, otherwise no event.
        """
        line, self.line = self.line, ''
        if not line:
            if not self.data:
                self.event = ''; return None
            event = dict(event=self.event or 'message', data='\n'.join(self.data))
            self.data = []; self.event = ''; self.count += 1
            if self.count > self.max_events: raise SSEProtocolError('event limit exceeded')
            return event
        if line.startswith(':'): return None
        field, sep, value = line.partition(':')
        if value.startswith(' '): value = value[1:]
        if field == 'data': self.data.append(value)
        elif field == 'event': self.event = value
        # id/retry/unknown fields do not alter this finite HTTP replay client.
        return None

    def feed(self, raw: bytes) -> list[SSEEvent]:
        """Decode a transport chunk, preserving split UTF-8 and CRLF state.

        Parameters
        ----------
        raw : bytes
            Next contiguous wire bytes; may contain zero or several events.

        Returns
        -------
        list[SSEEvent]
            Fully framed events, in arrival order; empty for partial frames.

        Raises
        ------
        SSEProtocolError
            If already closed or a cumulative limit is exceeded.
        UnicodeError
            If the wire contains malformed UTF-8.
        """
        if self.closed: raise SSEProtocolError('decoder already closed')
        if not isinstance(raw,bytes): raise TypeError('feed requires bytes, not decoded strings')
        self.bytes += len(raw)
        if self.bytes > self.max_bytes: raise SSEProtocolError('byte limit exceeded')
        text = self.decoder.decode(raw)
        events = []
        for char in text:
            if self.after_cr:
                self.after_cr = False
                if char == '\n': continue
            if char in '\r\n':
                event = self._line()
                if event is not None: events.append(event)
                self.after_cr = char == '\r'
            else: self.line += char
        return events

    def finish(self) -> bool:
        """Close the decoder without dispatching an unfinished EOF event.

        Returns
        -------
        bool
            Whether a buffered line or data field remains unfinished.

        Raises
        ------
        UnicodeError
            If EOF cuts an unfinished UTF-8 code point.
        """
        self.decoder.decode(b'',final=True)  # Incomplete UTF-8 is a protocol error.
        self.closed = True
        # WHATWG does not dispatch an unfinished event at EOF.
        return bool(self.line or self.data)


class ChatStream:
    """No reconnect/retry: partial text is evidence, never a completed answer."""
    def __init__(self) -> None:
        """Initialize a finite single-choice text-completion lifecycle.

        Returns
        -------
        None
            Empty content, clocks and event history; no network activity.
        """
        self.decoder = SSEDecoder(); self.events = []; self.content = ''
        self.finish_reason = None; self.usage = None; self.done = False
        self.first_event_s = None; self.first_content_s = None; self.last_s = 0.
        self.error = None; self.closed = False; self.cancelled = False; self.unfinished_frame = False

    def _event(self, event: SSEEvent, at_s: float) -> None:
        """Validate and reduce one already framed chat event.

        Parameters
        ----------
        event : SSEEvent
            Message event containing JSON or the literal [DONE].
        at_s : float
            Chunk-arrival wall seconds since request start.

        Returns
        -------
        None
            Updates content, usage and lifecycle; rejects out-of-order data.
        """
        if self.done: raise SSEProtocolError('data after DONE')
        if event['event'] != 'message': raise SSEProtocolError('only message events supported')
        if self.first_event_s is None: self.first_event_s = at_s
        body=event['data']
        row=dict(index=len(self.events),at_s=at_s,event=event['event'],data=body)
        self.events.append(row)
        if body == '[DONE]':
            if self.finish_reason is None: raise SSEProtocolError('DONE without finish_reason')
            self.done = True; row['kind']='done'; return
        obj=json.loads(body)
        if not isinstance(obj,dict): raise SSEProtocolError('JSON object required')
        if 'error' in obj: raise SSEProtocolError('server error event')
        choices=obj.get('choices')
        if not isinstance(choices,list): raise SSEProtocolError('choices list required')
        usage=obj.get('usage')
        if usage is not None:
            if not isinstance(usage,dict): raise SSEProtocolError('invalid usage')
            for key in ('prompt_tokens','completion_tokens','total_tokens'):
                if type(usage.get(key)) is not int or usage[key]<0: raise SSEProtocolError('invalid usage counts')
            if usage['total_tokens'] != usage['prompt_tokens']+usage['completion_tokens']:
                raise SSEProtocolError('usage total mismatch')
            self.usage=usage
        if not choices:
            if usage is None: raise SSEProtocolError('empty choices without usage')
            row['kind']='usage'; return
        if len(choices)!=1 or not isinstance(choices[0],dict) or type(choices[0].get('index')) is not int or choices[0]['index']!=0:
            raise SSEProtocolError('only object choice zero supported')
        choice=choices[0]; delta=choice.get('delta',{})
        if not isinstance(delta,dict): raise SSEProtocolError('delta object required')
        if any(key not in ('role','content') and value is not None for key,value in delta.items()):
            raise SSEProtocolError('unsupported non-text delta')
        if delta.get('role') not in (None,'assistant'): raise SSEProtocolError('unsupported role')
        text=delta.get('content')
        if text is not None and not isinstance(text,str): raise SSEProtocolError('content must be text or null')
        if text:
            if self.finish_reason is not None: raise SSEProtocolError('content after finish')
            if self.first_content_s is None: self.first_content_s=at_s
            self.content += text
        finish=choice.get('finish_reason')
        if finish is not None:
            if finish not in ('stop','length') or self.finish_reason is not None:
                raise SSEProtocolError('invalid or duplicate finish')
            self.finish_reason=finish
        row['kind']='finish' if finish is not None else ('content' if text else 'role_or_empty')

    def feed(self, raw: bytes, at_s: float) -> None:
        """Feed wire bytes with a finite monotonic client arrival clock.

        Parameters
        ----------
        raw : bytes
            Contiguous transport bytes, not decoded token strings.
        at_s : float
            Seconds since request start, not GPU or token-emission time.

        Returns
        -------
        None
            Updates state and preserves errors before raising them.
        """
        try:
            if self.closed: raise SSEProtocolError('stream already closed')
            if self.error: raise SSEProtocolError('stream already failed')
            if type(at_s) not in (int,float) or not math.isfinite(at_s) or at_s<self.last_s:
                raise SSEProtocolError('invalid or nonmonotonic arrival clock')
            self.last_s=at_s
            for event in self.decoder.feed(raw): self._event(event,at_s)
        except (ValueError,TypeError,UnicodeError) as exc:
            self.error=type(exc).__name__ + ': ' + str(exc)
            raise

    def finish(self, cancelled: bool = False, transport_error: str | None = None) -> ChatResult:
        """Close the stream and classify complete, partial or failed evidence.

        Parameters
        ----------
        cancelled : bool
            Whether this client deliberately stopped reading.
        transport_error : str or None
            Transport failure explanation, if observed.

        Returns
        -------
        ChatResult
            Content, lifecycle, usage and raw event history. Clocks are seconds;
            partial text is retained but is never a completed answer.

        Raises
        ------
        SSEProtocolError
            If this lifecycle was already closed.
        """
        if self.closed: raise SSEProtocolError('stream already closed')
        self.closed=True; self.cancelled=cancelled
        try: self.unfinished_frame=self.decoder.finish()
        except UnicodeError as exc: self.error=type(exc).__name__ + ': incomplete UTF-8'
        if transport_error: self.error=str(transport_error)
        if self.error: status='failed'
        elif cancelled: status='cancelled'
        elif self.done and self.finish_reason is not None and not self.unfinished_frame: status='complete'
        else: status='partial'
        return dict(status=status,content=self.content,finish_reason=self.finish_reason,done=self.done,
                    usage=self.usage,first_event_s=self.first_event_s,first_content_s=self.first_content_s,
                    wall_s=self.last_s,byte_count=self.decoder.bytes,event_count=len(self.events),
                    events=self.events,error=self.error,unfinished_frame=self.unfinished_frame,
                    clock='client chunk-arrival wall; not token emission or GPU time')


def synthetic_fixture() -> dict[str, Any]:
    """Construct and parse invented byte/clock examples without a server.

    Returns
    -------
    dict
        Request, frames, hex chunks and complete/partial/cancelled results.
        Usage counts and clocks are invented, not tokenizer measurements.
    """
    def chunk(delta: dict[str, str], finish: str | None = None) -> dict[str, Any]:
        """Build one illustrative OpenAI-style chunk envelope.

        Parameters
        ----------
        delta : dict[str, str]
            Role/content fields; empty for the finish event.
        finish : str or None
            Terminal reason, otherwise a nonterminal event.

        Returns
        -------
        dict
            A single-choice envelope; no tokenization is performed.
        """
        return dict(id='illustrative',object='chat.completion.chunk',model='topic04',
                    choices=[dict(index=0,delta=delta,finish_reason=finish)])
    payload=dict(model='topic04',messages=[dict(role='user',content='Say café ☕ briefly.')],
                 max_tokens=16,temperature=0,stream=True,stream_options=dict(include_usage=True))
    bodies=[chunk(dict(role='assistant',content='')),chunk(dict(content='Café ')),
            chunk(dict(content='☕.')),chunk({},'stop'),
            dict(choices=[],usage=dict(prompt_tokens=13,completion_tokens=4,total_tokens=17))]
    # The synthetic usage counts are invented, not tokenizer measurements.
    frames=[': heartbeat\r\n\r\n']+['data: '+json.dumps(b,ensure_ascii=False,separators=(',',':'))+'\r\n\r\n' for b in bodies]+['data: [DONE]\r\n\r\n']
    arrival=[.01,.03,.10,.14,.15,.16,.17]
    chunks=[]
    for text,at_s in zip(frames,arrival):
        raw=text.encode('utf-8')
        # Always split CRLF; split UTF-8 continuation sequences when present.
        cuts=sorted(set([len(raw)-3]+[i+1 for i,b in enumerate(raw) if b in (0xc3,0xe2)]))
        previous=0
        for end in cuts+[len(raw)]:
            chunks.append(dict(at_s=at_s,hex=raw[previous:end].hex())); previous=end
    stream=ChatStream()
    for c in chunks: stream.feed(bytes.fromhex(c['hex']),c['at_s'])
    complete=stream.finish()
    prefix=ChatStream()
    for c in chunks:
        if c['at_s']>.10: break
        prefix.feed(bytes.fromhex(c['hex']),c['at_s'])
    partial=prefix.finish()
    cancelled=ChatStream()
    for c in chunks:
        if c['at_s']>.10: break
        cancelled.feed(bytes.fromhex(c['hex']),c['at_s'])
    return dict(schema='topic04-sse-fixture-v1',kind='illustrative bytes, clocks and usage; no server or tokenizer executed',
                request=payload,frames=frames,chunks=chunks,complete=complete,partial=partial,
                cancelled=cancelled.finish(cancelled=True),source='https://html.spec.whatwg.org/multipage/server-sent-events.html')

"""Canonical bounded paired smoke. Import is CPU-only; GPU work starts at run_lab()."""
import gc
import json
import math
import os
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MODEL_ID='Qwen/Qwen2.5-0.5B-Instruct'
REVISION='7ae557604adf67be50417f59c2c2f167def9a775'
PROTOCOL='topic-03-paired-smoke-v3'
TRIALS, MAX_NEW=3,24
PROMPTS=[
    dict(bucket='format',text='Reply with exactly one word: blue or orange. What color is the sky on a clear day?'),
    dict(bucket='arithmetic',text='Compute 17 + 28. Give only the integer.'),
    dict(bucket='instruction',text='Write exactly three comma-separated verbs for measuring a service.'),
    dict(bucket='reasoning',text='A server has 2 GiB free and a new request needs 3 GiB. In one sentence, say whether it fits and why.'),
]


def checkpoint(path: str | Path, data: dict[str, Any]) -> None:
    """Atomically replace one attempt's progress checkpoint.

    Parameters
    ----------
    path : str or Path
        File inside the caller's fresh attempt directory.
    data : dict
        JSON-compatible progress/results; nonfinite values are rejected.

    Returns
    -------
    None
        Writes UTF-8 JSON via a sibling temporary file; no cache changes.
    """
    path=Path(path); temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def resource_gate(require_idle: bool = False) -> dict[str, float | int]:
    """Read the unchanged historical operator resource envelope.

    Parameters
    ----------
    require_idle : bool
        Reject an existing GPU compute process before model startup.

    Returns
    -------
    dict
        available_ram_gib and device_used_mib, in GiB and device-wide MiB.

    Raises
    ------
    RuntimeError
        Unsupported host, less than 24 GiB RAM, over 12 GiB device use,
        or unsafe ownership. This is NOT the current student pilot profile.
    """
    mem=Path('/proc/meminfo')
    if not mem.exists(): raise RuntimeError('Linux/WSL RAM gate unavailable; use supported CUDA runtime')
    available=int(next(line.split()[1] for line in mem.read_text().splitlines() if line.startswith('MemAvailable:')))*1024
    used=[int(x) for x in subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True,timeout=10).split()]
    if available < 24*2**30 or not used or max(used)>12288: raise RuntimeError('STOP: RAM/GPU resource gate')
    if require_idle:
        active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=10).strip()
        if active: raise RuntimeError('STOP: GPU compute process already active')
    return dict(available_ram_gib=available/2**30,device_used_mib=max(used))


def resolve_snapshot() -> Path:
    """Resolve the exact 0.5B snapshot, reusing persistent cache first.

    Returns
    -------
    Path
        Local immutable snapshot directory. In offline mode a missing pinned
        snapshot raises RuntimeError instead of silently downloading.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    cache=Path(os.environ.get('LSM_MODEL_CACHE',Path.cwd()/'.cache/models')).expanduser()
    try: snapshot=snapshot_download(MODEL_ID,revision=REVISION,cache_dir=str(cache),local_files_only=True)
    except LocalEntryNotFoundError:
        if os.environ.get('HF_HUB_OFFLINE')=='1': raise RuntimeError('Pinned offline snapshot missing; do not redownload')
        snapshot=snapshot_download(MODEL_ID,revision=REVISION,cache_dir=str(cache))
    return Path(snapshot)



"""Proposed bounded student profile; browser validation remains mandatory.

This does not change the judge or operator gates. Budgets are a starting profile
for sequential 0.5B experiments, not universal package requirements.
"""
from pathlib import Path
import subprocess
from typing import Mapping, Sequence, TypedDict


class StudentResources(TypedDict):
    """Observed headroom, not an allocation guarantee."""

    available_ram_gib: float
    device_used_mib: int
    device_free_mib: int
    profile: str
    idle_required: bool
    active_compute: bool


def assess_student_resources(
    available_ram_bytes: int, gpu_total_mib: int, gpu_used_mib: int,
    active_compute: bool, require_idle: bool = False,
) -> StudentResources:
    """Apply the proposed sequential small-model resource budget.

    Parameters
    ----------
    available_ram_bytes : int
        MemAvailable, in bytes; not total installed RAM or VRAM.
    gpu_total_mib : int
        Total memory of the single selected GPU, in MiB.
    gpu_used_mib : int
        Device-wide used memory, in MiB; not PyTorch allocated memory.
    active_compute : bool
        Whether nvidia-smi lists a compute process on this GPU.
    require_idle : bool, default False
        Startup requires no compute process and 5 GiB host/device headroom.
        During execution retain 2 GiB host and 512 MiB device headroom.

    Returns
    -------
    StudentResources
        Observed RAM GiB, GPU used/free MiB and profile identifier.

    Raises
    ------
    ValueError
        If memory counters are negative or used exceeds total.
    RuntimeError
        If the proposed budget fails or startup would overlap another owner.

    Notes
    -----
    These are provisional stop rules, not certified free-Colab minima, an OS
    sandbox or proof that loading will succeed. Do not kill unknown processes.
    """
    if min(available_ram_bytes, gpu_used_mib) < 0 or gpu_total_mib <= 0 or gpu_used_mib > gpu_total_mib:
        raise ValueError('Invalid observed memory counters')
    ram_gib = available_ram_bytes / 2**30
    free_mib = gpu_total_mib - gpu_used_mib
    host_floor = 5 if require_idle else 2
    gpu_floor = 5120 if require_idle else 512
    observed = f'available host RAM={ram_gib:.2f} GiB; GPU free={free_mib} MiB'
    if require_idle and active_compute:
        raise RuntimeError('STOP: a GPU compute process is active. Stop only your own previous run; '
                           'do not terminate another owner. ' + observed)
    if ram_gib < host_floor or free_mib < gpu_floor:
        raise RuntimeError(f'STOP: student small-model budget requires host >= {host_floor} GiB '
                           f'and device free >= {gpu_floor} MiB. {observed}. Preserve partial '
                           'evidence; inspect your own memory use or use the recorded CPU route '
                           '(not GPU completion). Do not retry the whole experiment blindly.')
    return dict(available_ram_gib=ram_gib, device_used_mib=gpu_used_mib,
                device_free_mib=free_mib, profile='student-small-model-provisional-v1',
                idle_required=require_idle, active_compute=active_compute)


def validate_student_resource_samples(samples: Sequence[Mapping[str, object]]) -> bool:
    """Check the explicit student startup and teardown resource records.

    Parameters
    ----------
    samples : sequence of mappings
        Two idle snapshots in startup/teardown order. RAM is in GiB; GPU
        counters are device-wide MiB, not allocator bytes.

    Returns
    -------
    bool
        True if both records satisfy the provisional profile and recovery.
        This validates recorded fields, not authenticity or free-tier access.

    Raises
    ------
    ValueError
        If the profile, idle observations, counters or recovery disagree.
    """
    if len(samples) != 2:
        raise ValueError('Student service needs startup and teardown samples')
    for sample in samples:
        ram, used, free = (sample.get(k) for k in ('available_ram_gib','device_used_mib','device_free_mib'))
        if (sample.get('profile') != 'student-small-model-provisional-v1'
                or sample.get('idle_required') is not True or sample.get('active_compute') is not False
                or type(ram) not in (int,float) or not 5 <= ram < float('inf')
                or type(used) is not int or used < 0 or type(free) is not int or free < 5120):
            raise ValueError('Invalid provisional student idle resource record')
    if samples[-1]['device_used_mib'] > samples[0]['device_used_mib'] + 64:
        raise ValueError('Student device memory did not recover after teardown')
    return True


def student_resource_gate(require_idle: bool = False) -> StudentResources:
    """Read Linux and nvidia-smi counters without importing GPU libraries.

    Parameters
    ----------
    require_idle : bool, default False
        Apply the startup ownership/headroom checks when True.

    Returns
    -------
    StudentResources
        Current counters and proposed student profile, not a runtime receipt.

    Raises
    ------
    RuntimeError
        If Linux/CUDA counters are unavailable, multiple GPUs are ambiguous,
        startup ownership is unsafe, or the proposed budget fails.
    """
    meminfo = Path('/proc/meminfo')
    if not meminfo.exists():
        raise RuntimeError('Linux memory counters unavailable. Use a Colab GPU runtime for GPU work; '
                           'the CPU learning route remains available.')
    try:
        available = int(next(line.split()[1] for line in meminfo.read_text().splitlines()
                             if line.startswith('MemAvailable:'))) * 1024
        rows = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.total,memory.used',
             '--format=csv,noheader,nounits'], text=True, timeout=10).strip().splitlines()
        if len(rows) != 1:
            raise RuntimeError('This student profile needs one unambiguous GPU, not zero/multiple GPUs.')
        total, used = (int(part.strip()) for part in rows[0].split(','))
        active = bool(subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
            text=True, timeout=10).strip()) if require_idle else False
    except (FileNotFoundError, subprocess.SubprocessError, StopIteration, ValueError) as exc:
        raise RuntimeError('CUDA counters unavailable. In Colab choose a GPU runtime, reconnect '
                           'and inspect its resources; do not call CPU execution GPU evidence.') from exc
    return assess_student_resources(available, total, used, active, require_idle)

resource_gate = student_resource_gate
"""Bounded local API contract rehearsal. No GPU/import/network work at module import."""
import atexit
import json
import os
import platform
import signal
import socket
import subprocess
import sys
import time
import threading
from contextlib import contextmanager
from datetime import datetime,timezone
from pathlib import Path
from types import FrameType
from typing import Any, Callable, Iterator

PROTOCOL='topic-04-api-smoke-v3'
SERVED_NAME='topic04'
PROMPT='Reply with one short sentence explaining what an API server does.'
RAW_PROMPT='Say hello briefly.'


@contextmanager
def request_deadline(seconds: float = 120) -> Iterator[None]:
    """POSIX/main-thread absolute wall deadline, distinct from HTTP read inactivity.

    Refuse an existing real-time alarm rather than overwrite another owner's timer.
    Always restore the previous handler. The course runtime is Linux/WSL/Colab.

    Parameters
    ----------
    seconds : float
        Whole-request wall deadline, in seconds; not a read-inactivity timeout.

    Yields
    ------
    None
        Protected request/iteration scope, restoring the prior alarm handler.

    Raises
    ------
    TimeoutError
        The absolute wall deadline expires, even when reads keep progressing.
    """
    if threading.current_thread() is not threading.main_thread() or not hasattr(signal, 'setitimer'):
        raise RuntimeError('Absolute request deadline requires POSIX main thread')
    if signal.getitimer(signal.ITIMER_REAL)[0]:
        raise RuntimeError('An existing alarm owns this process; use an isolated runtime')
    previous = signal.getsignal(signal.SIGALRM)
    def expire(signum: int, frame: FrameType | None) -> None:
        """Propagate expiry from the owned POSIX alarm.

        Parameters
        ----------
        signum : int
            Delivered signal number.
        frame : FrameType or None
            Interrupted Python stack frame; not modified.

        Returns
        -------
        None
            Always raises TimeoutError; never suppresses the deadline.
        """
        raise TimeoutError('Absolute request deadline exceeded')
    signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def normalize_input_ids(encoded: Any) -> list[int]:
    """Normalize one tokenizer result across supported Transformers versions.

    Parameters
    ----------
    encoded : tokenizer output
        One list of IDs or a BatchEncoding containing input_ids.

    Returns
    -------
    list[int]
        One nonempty sequence of nonnegative integer IDs, shape [S].

    Raises
    ------
    ValueError
        A batch, empty sequence or invalid ID type is supplied.
    """
    value=encoded['input_ids'] if hasattr(encoded,'keys') else encoded
    if isinstance(value,list) and len(value)==1 and isinstance(value[0],list): value=value[0]
    if not isinstance(value,list) or not value or any(type(x) is not int or x<0 for x in value):
        raise ValueError('Expected one nonempty integer input_ids sequence')
    return value


def stop_owned_process_group(server: subprocess.Popen, term_seconds: float = 20, kill_seconds: float = 10) -> None:
    """Stop only a POSIX session created by this experiment.

    Parameters
    ----------
    server : subprocess.Popen
        Owned process created with start_new_session=True, not an arbitrary PID.
    term_seconds : float
        Graceful group-exit budget in seconds.
    kill_seconds : float
        Forced-exit and verification budget in seconds.

    Returns
    -------
    None
        Owned descendants are gone; failure raises instead of claiming cleanup.
    """
    try: os.killpg(server.pid,signal.SIGTERM)
    except ProcessLookupError: pass
    deadline=time.monotonic()+term_seconds
    while time.monotonic()<deadline:
        server.poll()
        try: os.killpg(server.pid,0)
        except ProcessLookupError: break
        time.sleep(.1)
    else:
        try: os.killpg(server.pid,signal.SIGKILL)
        except ProcessLookupError: pass
    server.wait(timeout=kill_seconds)
    deadline=time.monotonic()+kill_seconds
    while time.monotonic()<deadline:
        try: os.killpg(server.pid,0)
        except ProcessLookupError: return
        time.sleep(.1)
    raise RuntimeError('Owned process group remains after teardown')


def run_lab(snapshot: str | Path, output_path: str | Path,
            task_probe: Callable[[Any, Any], dict[str, Any]] | None = None) -> dict[str, Any]:
    """Run one bounded local service lifetime and preserve its raw evidence.

    Parameters
    ----------
    snapshot : str or Path
        Already resolved immutable 0.5B snapshot, loaded offline.
    output_path : str or Path
        New checkpoint path; never overwrites an earlier attempt.
    task_probe : callable or None
        Optional auxiliary probe using the owned SDK client and tokenizer.

    Returns
    -------
    dict
        Service identity, prompts, raw SSE/chunks, probes, resource snapshots
        and verified teardown. All wall clocks are seconds, not kernel times.

    Raises
    ------
    RuntimeError
        Startup, resource or service contract fails; partial JSON is retained.
    """
    import httpx,openai,torch,vllm,transformers
    from transformers import AutoTokenizer
    from importlib.metadata import version
    if version('vllm')!='0.30.0+cu129' or vllm.__version__!='0.30.0': raise RuntimeError('Pinned vLLM 0.30.0+cu129 required')
    output_path=Path(output_path)
    if output_path.exists(): raise FileExistsError('Preserve prior evidence: choose a fresh output path')
    output_path.parent.mkdir(parents=True,exist_ok=True)
    preflight=resource_gate(require_idle=True)
    tokenizer=AutoTokenizer.from_pretrained(str(snapshot),local_files_only=True)
    def prompt_record(text: str) -> dict[str, Any]:
        """Record the literal chat, rendered template and prompt IDs.

        Parameters
        ----------
        text : str
            Literal user message for this controlled request.

        Returns
        -------
        dict
            messages, rendered_chat, input_token_ids/pieces and prompt_tokens.
        """
        messages=[dict(role='user',content=text)]
        ids=normalize_input_ids(tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True))
        return dict(messages=messages,rendered_chat=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True),
                    input_token_ids=ids,prompt_tokens=len(ids),input_token_pieces=tokenizer.convert_ids_to_tokens(ids))
    base='http://127.0.0.1:8000/v1'
    with socket.socket() as probe:
        if probe.connect_ex(('127.0.0.1',8000))==0: raise RuntimeError('Port 8000 occupied; do not stop an unowned server')
    args=['serve',str(snapshot),'--served-model-name',SERVED_NAME,'--host','127.0.0.1','--port','8000',
          '--dtype','float16','--max-model-len','2048','--max-num-seqs','4','--gpu-memory-utilization','0.45','--enforce-eager']
    is_wsl=bool(os.environ.get('WSL_INTEROP')) or 'microsoft' in platform.release().lower()
    env=os.environ.copy(); env['HF_HUB_OFFLINE']='1'; env['TRANSFORMERS_OFFLINE']='1'
    if is_wsl: env['VLLM_USE_V2_MODEL_RUNNER']='0'; env['VLLM_USE_FLASHINFER_SAMPLER']='0'
    data=dict(contract=PROTOCOL,status='running',model_id=MODEL_ID,revision=REVISION,served_name=SERVED_NAME,
              utc_started=datetime.now(timezone.utc).isoformat(),vllm=version('vllm'),vllm_module=vllm.__version__,openai_sdk=openai.__version__,
              httpx=httpx.__version__,transformers=transformers.__version__,
              torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),
              launch_args=['serve','<pinned-local-snapshot>']+args[2:],wsl_v1_runner_fallback=is_wsl,
              resource_samples=[preflight],request_prompt=prompt_record(PROMPT),raw_prompt=prompt_record(RAW_PROMPT),
              limits=dict(max_model_len=2048,max_num_seqs=4,gpu_memory_utilization=.45),
              limitations=['single sequential API smoke, not throughput benchmark','first call may include lazy runtime initialization',
                           'TCP/HTTP chunk boundaries are not generated token boundaries','client disconnect reclamation not measured'])
    checkpoint(output_path,data)
    # Calibration and held-out are sequential owned lifetimes in ONE attempt.
    # Keep separate immutable logs instead of colliding at server.local.log.
    log=output_path.with_suffix('.server.local.log').open('x',encoding='utf-8')
    server=None; stopped=False; client=None
    def stop_server() -> None:
        """Idempotently stop this lifetime's owned group and close its log.

        Returns
        -------
        None
            No unrelated processes are terminated; cleanup failure propagates.
        """
        nonlocal stopped
        if stopped: return
        stopped=True
        if server is not None:
            stop_owned_process_group(server)
        log.close()
    atexit.register(stop_server)
    try:
        server=subprocess.Popen([sys.executable,'-m','vllm.entrypoints.cli.main']+args,
                                env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        begin=time.monotonic(); deadline=begin+300
        while time.monotonic()<deadline:
            if server.poll() is not None: raise RuntimeError('vLLM exited during startup; inspect private local log')
            resource_gate()
            try:
                response=httpx.get(base+'/models',timeout=2,trust_env=False)
                if response.status_code==200:
                    advertised=[x['id'] for x in response.json()['data']]
                    if SERVED_NAME not in advertised: raise RuntimeError('served name mismatch')
                    data['advertised_models']=advertised; break
            except httpx.RequestError: pass
            time.sleep(2)
        else: raise TimeoutError('Startup exceeded 300 s')
        data['startup_s']=time.monotonic()-begin; checkpoint(output_path,data)
        client=openai.OpenAI(base_url=base,api_key='EMPTY',timeout=120,max_retries=0,
                            http_client=httpx.Client(trust_env=False))
        request=dict(model=SERVED_NAME,messages=[dict(role='user',content=PROMPT)],max_tokens=48,temperature=0)
        start=time.perf_counter()
        with request_deadline(): answer=client.chat.completions.create(**request)
        data['nonstream']=dict(request=request,wall_s=time.perf_counter()-start,finish_reason=answer.choices[0].finish_reason,
                               usage=answer.usage.model_dump() if answer.usage else None,content=answer.choices[0].message.content)
        if not data['nonstream']['content']: raise RuntimeError('Empty nonstream response')
        checkpoint(output_path,data); resource_gate()
        start=time.perf_counter(); first=None; pieces=[]; finish=None; chunks=0
        # Cover creation AND iteration: progressing reads must not reset wall time.
        sdk_deadline=request_deadline(); sdk_deadline.__enter__()
        stream=None
        try:
            stream=client.chat.completions.create(**request,stream=True)
            for chunk in stream:
                chunks+=1
                if chunks>512: raise RuntimeError('SDK chunk budget exceeded')
                delta=chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    if first is None: first=time.perf_counter()-start
                    pieces.append(delta)
                if chunk.choices and chunk.choices[0].finish_reason: finish=chunk.choices[0].finish_reason
        finally:
            try:
                if stream is not None: stream.close()
            finally: sdk_deadline.__exit__(None,None,None)
        data['sdk_stream']=dict(first_content_s=first,wall_s=time.perf_counter()-start,finish_reason=finish,
                                content=''.join(pieces),sdk_chunks=chunks)
        if first is None or finish is None: raise RuntimeError('Incomplete SDK stream')
        checkpoint(output_path,data); resource_gate()
        payload=dict(model=SERVED_NAME,messages=[dict(role='user',content=RAW_PROMPT)],max_tokens=16,temperature=0,
                     stream=True,stream_options=dict(include_usage=True))
        parser=ChatStream(); raw_chunks=[]; start=time.perf_counter()
        with request_deadline(), httpx.stream('POST',base+'/chat/completions',json=payload,timeout=120,trust_env=False) as response:
            response.raise_for_status()
            media=response.headers.get('content-type','').split(';')[0].lower()
            if media!='text/event-stream': raise RuntimeError('Wrong SSE content type')
            for raw in response.iter_bytes():
                at=time.perf_counter()-start
                raw_chunks.append(dict(at_s=at,hex=raw.hex()))
                parser.feed(raw,at)
        record=parser.finish(); record['request']=payload; record['http_status']=response.status_code
        record['content_type']=media; record['chunks']=raw_chunks
        data['raw_sse']=record; checkpoint(output_path,data)
        if record['status']!='complete' or not record['content']: raise RuntimeError('Incomplete raw SSE')
        negative={}
        try:
            with request_deadline(): client.chat.completions.create(model='not-the-served-model',messages=[dict(role='user',content='hi')],max_tokens=1)
        except openai.APIStatusError as exc: negative['wrong_model']=dict(status=exc.status_code)
        long_text=' blue'*2300
        rendered=normalize_input_ids(tokenizer.apply_chat_template([dict(role='user',content=long_text)],tokenize=True,add_generation_prompt=True))
        if not 2048<len(rendered)<8192: raise RuntimeError('Bounded over-context fixture invalid')
        try:
            with request_deadline(): client.chat.completions.create(model=SERVED_NAME,messages=[dict(role='user',content=long_text)],max_tokens=16)
        except openai.APIStatusError as exc: negative['over_context']=dict(status=exc.status_code,rendered_prompt_tokens=len(rendered))
        if set(negative)!={'wrong_model','over_context'} or negative['wrong_model']['status']!=404 or negative['over_context']['status']!=400:
            raise RuntimeError('Expected wrong-model 404 / over-context 400 contract not met')
        if task_probe is not None:
            with request_deadline(): data['task_probe']=task_probe(client,tokenizer)
        data['timeout_policy']=dict(http_phase_s=120,absolute_request_s=120,mechanism='POSIX ITIMER_REAL main thread')
        data['negative']=negative; data['status']='complete'; checkpoint(output_path,data)
    except BaseException as exc:
        data['status']='failed'; data['error_type']=type(exc).__name__; raise
    finally:
        try:
            if client is not None: client.close()
            stop_server(); atexit.unregister(stop_server)
            data['teardown']=dict(owned_process_group_stopped=True,server_returncode=None if server is None else server.returncode)
            data['resource_samples'].append(resource_gate(require_idle=True))
        except BaseException as exc:
            data['status']='failed'; data['cleanup_error_type']=type(exc).__name__
            raise
        finally:
            data['utc_finished']=datetime.now(timezone.utc).isoformat(); checkpoint(output_path,data)
    return data


def validate_artifact(data: dict[str, Any], *, student_resources: bool = False) -> bool:
    """Validate a completed service artifact with an explicitly chosen envelope.

    Parameters
    ----------
    data : dict
        Runtime/probe/raw-SSE/resource/teardown JSON from the owned service.
    student_resources : bool
        Require the explicit provisional student profile when True. False
        preserves the original 24 GiB operator envelope and historical evidence.

    Returns
    -------
    bool
        True for consistent evidence; not a speed, quality or authenticity claim.

    Raises
    ------
    ValueError
        Any request, lifecycle, count, clock, resource or cleanup check fails.
    """
    import math
    def require(value: object, message: str) -> None:
        """Reject one failed consistency condition.

        Parameters
        ----------
        value : object
            Condition to test, not an untrusted executable expression.
        message : str
            Diagnostic for the violated field.

        Returns
        -------
        None
            False conditions raise ValueError.
        """
        if not value: raise ValueError(message)
    def finite_number(value: object) -> bool:
        """Recognize finite JSON numeric fields, excluding bools.

        Parameters
        ----------
        value : object
            Untrusted counter or clock value.

        Returns
        -------
        bool
            True only for finite built-in int or float values.
        """
        return type(value) in (int,float) and math.isfinite(value)
    def usage_counts(usage: Any) -> bool:
        """Check token-count fields of a third-party usage envelope.

        Parameters
        ----------
        usage : object
            Candidate SDK/SSE usage mapping, possibly absent or malformed.

        Returns
        -------
        bool
            True for nonnegative integer counts and a positive output count.
        """
        return isinstance(usage,dict) and all(type(usage.get(k)) is int and usage[k]>=0
            for k in ('prompt_tokens','completion_tokens','total_tokens')) and usage['completion_tokens']>0
    require(data.get('contract')==PROTOCOL and data.get('status')=='complete','incomplete/wrong protocol')
    require(data.get('revision')==REVISION and data.get('model_id')==MODEL_ID,'model identity')
    require(data.get('vllm')=='0.30.0+cu129' and data.get('vllm_module')=='0.30.0' and all(isinstance(data.get(key),str) and data[key] for key in ('openai_sdk','httpx','transformers','torch','cuda','gpu')),'runtime pins/identity')
    require(data.get('served_name')==SERVED_NAME and SERVED_NAME in data.get('advertised_models',[]),'served identity')
    expected_limits=dict(max_model_len=2048,max_num_seqs=4,gpu_memory_utilization=.45)
    require(data.get('limits')==expected_limits,'fixed bounded launch limits')
    require(type(data['limits']['max_model_len']) is int and type(data['limits']['max_num_seqs']) is int and
            finite_number(data['limits']['gpu_memory_utilization']),'numeric launch limit types')
    expected_launch=['serve','<pinned-local-snapshot>','--served-model-name',SERVED_NAME,'--host','127.0.0.1','--port','8000',
                     '--dtype','float16','--max-model-len','2048','--max-num-seqs','4','--gpu-memory-utilization','0.45','--enforce-eager']
    require(data.get('launch_args')==expected_launch,'exact loopback bounded launch')
    require(type(data.get('wsl_v1_runner_fallback')) is bool,'explicit WSL fallback')
    require(finite_number(data.get('startup_s')) and 0<data['startup_s']<=300,'bounded startup clock')
    expected_request=dict(model=SERVED_NAME,messages=[dict(role='user',content=PROMPT)],max_tokens=48,temperature=0)
    require(data['nonstream'].get('request')==expected_request,'exact nonstream request envelope')
    require(type(data['nonstream']['request']['max_tokens']) is int and
            finite_number(data['nonstream']['request']['temperature']),'nonstream envelope numeric types')
    for key,text in [('request_prompt',PROMPT),('raw_prompt',RAW_PROMPT)]:
        prompt=data[key]
        require(prompt['messages']==[dict(role='user',content=text)],'literal request')
        require(prompt['prompt_tokens']==len(prompt['input_token_ids'])==len(prompt['input_token_pieces'])>0,'prompt count')
        require(all(type(x) is int and x>=0 for x in prompt['input_token_ids']),'ID types')
    for key in ('nonstream','sdk_stream'):
        result=data[key]
        require(isinstance(result.get('content'),str) and bool(result['content']),'content required')
        require(result.get('finish_reason') in ('stop','length'),'finish reason')
        require(finite_number(result['wall_s']) and result['wall_s']>0,'wall clock')
    require(finite_number(data['sdk_stream']['first_content_s']) and 0<data['sdk_stream']['first_content_s']<=data['sdk_stream']['wall_s'],'first content clock')
    raw=data['raw_sse']; replay=ChatStream()
    for chunk in raw['chunks']: replay.feed(bytes.fromhex(chunk['hex']),chunk['at_s'])
    reconstructed=replay.finish()
    require(all(raw[key]==value for key,value in reconstructed.items()),'raw byte replay mismatch')
    require(raw['http_status']==200 and raw['content_type']=='text/event-stream','SSE HTTP boundary')
    require(raw['status']=='complete' and raw['done'] and raw['content'],'partial stream is not success')
    expected_raw=dict(model=SERVED_NAME,messages=[dict(role='user',content=RAW_PROMPT)],max_tokens=16,temperature=0,
                      stream=True,stream_options=dict(include_usage=True))
    require(raw.get('request')==expected_raw,'exact raw request envelope')
    require(type(raw['request']['max_tokens']) is int and finite_number(raw['request']['temperature']) and
            raw['request']['stream'] is True and raw['request']['stream_options']['include_usage'] is True,'raw envelope types')
    require(raw['request']['messages']==data['raw_prompt']['messages'],'raw prompt identity')
    require(usage_counts(raw['usage']) and raw['usage']['prompt_tokens']==data['raw_prompt']['prompt_tokens'],'raw usage vs tokenizer')
    usage=data['nonstream']['usage']
    require(usage_counts(usage) and usage['prompt_tokens']==data['request_prompt']['prompt_tokens'] and usage['total_tokens']==usage['prompt_tokens']+usage['completion_tokens'],'nonstream usage')
    require(data['teardown'].get('owned_process_group_stopped') is True and type(data['teardown'].get('server_returncode')) is int,'owned server teardown')
    resources=data.get('resource_samples',[])
    if student_resources:
        require(validate_student_resource_samples(resources),'explicit provisional student resource envelope')
    else:
        require(len(resources)==2 and all(finite_number(r.get('available_ram_gib')) and r['available_ram_gib']>=24 and
                    type(r.get('device_used_mib')) is int and 0<=r['device_used_mib']<=12288 for r in resources),'operator resource envelope')
    require(resources[-1]['device_used_mib']<=resources[0]['device_used_mib']+64,'device allocation recovered after teardown')
    require(set(data['negative'])=={'wrong_model','over_context'},'negative probes')
    require(all(type(row['status']) is int for row in data['negative'].values()) and
            data['negative']['wrong_model']['status']==404 and data['negative']['over_context']['status']==400,
            'wrong-model must be 404; over-context must be 400, not arbitrary 4xx')
    require(type(data['negative']['over_context']['rendered_prompt_tokens']) is int and
            2048<data['negative']['over_context']['rendered_prompt_tokens']<8192,'bounded token rejection')
    return True

operator_validate_artifact = validate_artifact
def validate_artifact(data: dict) -> bool:
    """Validate the unchanged service workload with the explicit student profile.

    Parameters
    ----------
    data : dict
        Service JSON with startup/teardown profile, request and raw SSE records.

    Returns
    -------
    bool
        True for consistent completed evidence; not proof of authenticity.

    Raises
    ------
    ValueError
        If any service or provisional resource field violates its contract.
    """
    return operator_validate_artifact(data, student_resources=True)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Current Topic04 student-profile CPU contract check')
    parser.add_argument('artifact',type=Path)
    args = parser.parse_args()
    validate_artifact(json.loads(args.artifact.read_text()))
    print('Topic04 current artifact consistent; not task quality, GPU authenticity or judge acceptance')
