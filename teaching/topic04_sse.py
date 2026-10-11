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
