"""Bounded UTF-8/SSE decoder + single-choice chat lifecycle, standard library only.

Framing follows WHATWG event streams (CR/LF/CRLF, comments, multiline data).
Chat completion is a separate, deliberately strict course contract, not all SSE.
"""
import codecs
import json
import math


class SSEProtocolError(ValueError):
    pass


class SSEDecoder:
    def __init__(self, max_bytes=262144, max_events=4096):
        self.decoder = codecs.getincrementaldecoder('utf-8-sig')('strict')
        self.line = ''; self.data = []; self.event = ''; self.after_cr = False
        self.bytes = 0; self.count = 0; self.max_bytes = max_bytes; self.max_events = max_events
        self.closed = False

    def _line(self):
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

    def feed(self, raw):
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

    def finish(self):
        self.decoder.decode(b'',final=True)  # Incomplete UTF-8 is a protocol error.
        self.closed = True
        # WHATWG does not dispatch an unfinished event at EOF.
        return bool(self.line or self.data)


class ChatStream:
    """No reconnect/retry: partial text is evidence, never a completed answer."""
    def __init__(self):
        self.decoder = SSEDecoder(); self.events = []; self.content = ''
        self.finish_reason = None; self.usage = None; self.done = False
        self.first_event_s = None; self.first_content_s = None; self.last_s = 0.
        self.error = None; self.closed = False; self.cancelled = False; self.unfinished_frame = False

    def _event(self,event,at_s):
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

    def feed(self,raw,at_s):
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

    def finish(self,cancelled=False,transport_error=None):
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


def synthetic_fixture():
    def chunk(delta,finish=None):
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


def checkpoint(path,data):
    path=Path(path); temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def resource_gate(require_idle=False):
    mem=Path('/proc/meminfo')
    if not mem.exists(): raise RuntimeError('Linux/WSL RAM gate unavailable; use supported CUDA runtime')
    available=int(next(line.split()[1] for line in mem.read_text().splitlines() if line.startswith('MemAvailable:')))*1024
    used=[int(x) for x in subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True,timeout=10).split()]
    if available < 24*2**30 or not used or max(used)>12288: raise RuntimeError('STOP: RAM/GPU resource gate')
    if require_idle:
        active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=10).strip()
        if active: raise RuntimeError('STOP: GPU compute process already active')
    return dict(available_ram_gib=available/2**30,device_used_mib=max(used))


def resolve_snapshot():
    from huggingface_hub import snapshot_download
    cache=Path(os.environ.get('LSM_MODEL_CACHE',Path.cwd()/'.cache/models')).expanduser()
    try: snapshot=snapshot_download(MODEL_ID,revision=REVISION,cache_dir=str(cache),local_files_only=True)
    except FileNotFoundError:
        if os.environ.get('HF_HUB_OFFLINE')=='1': raise RuntimeError('Pinned offline snapshot missing; do not redownload')
        snapshot=snapshot_download(MODEL_ID,revision=REVISION,cache_dir=str(cache))
    return Path(snapshot)



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
from datetime import datetime,timezone
from pathlib import Path

PROTOCOL='topic-04-api-smoke-v3'
SERVED_NAME='topic04'
PROMPT='Reply with one short sentence explaining what an API server does.'
RAW_PROMPT='Say hello briefly.'


def normalize_input_ids(encoded):
    """Transformers 4/5 may return IDs or a BatchEncoding; accept one sequence only."""
    value=encoded['input_ids'] if hasattr(encoded,'keys') else encoded
    if isinstance(value,list) and len(value)==1 and isinstance(value[0],list): value=value[0]
    if not isinstance(value,list) or not value or any(type(x) is not int or x<0 for x in value):
        raise ValueError('Expected one nonempty integer input_ids sequence')
    return value


def stop_owned_process_group(server,term_seconds=20,kill_seconds=10):
    """Stop only a POSIX session we created, including worker descendants."""
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


def run_lab(snapshot,output_path):
    import httpx,openai,torch,vllm,transformers
    from transformers import AutoTokenizer
    if vllm.__version__!='0.29.0': raise RuntimeError('Pinned vLLM 0.29.0 required')
    output_path=Path(output_path)
    if output_path.exists(): raise FileExistsError('Preserve prior evidence: choose a fresh output path')
    output_path.parent.mkdir(parents=True,exist_ok=True)
    preflight=resource_gate(require_idle=True)
    tokenizer=AutoTokenizer.from_pretrained(str(snapshot),local_files_only=True)
    def prompt_record(text):
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
              utc_started=datetime.now(timezone.utc).isoformat(),vllm=vllm.__version__,openai_sdk=openai.__version__,
              httpx=httpx.__version__,transformers=transformers.__version__,
              torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),
              launch_args=['serve','<pinned-local-snapshot>']+args[2:],wsl_v1_runner_fallback=is_wsl,
              resource_samples=[preflight],request_prompt=prompt_record(PROMPT),raw_prompt=prompt_record(RAW_PROMPT),
              limits=dict(max_model_len=2048,max_num_seqs=4,gpu_memory_utilization=.45),
              limitations=['single sequential API smoke, not throughput benchmark','first call may include lazy runtime initialization',
                           'TCP/HTTP chunk boundaries are not generated token boundaries','client disconnect reclamation not measured'])
    checkpoint(output_path,data)
    log=output_path.with_name('server.local.log').open('x',encoding='utf-8')
    server=None; stopped=False; client=None
    def stop_server():
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
        start=time.perf_counter(); answer=client.chat.completions.create(**request)
        data['nonstream']=dict(request=request,wall_s=time.perf_counter()-start,finish_reason=answer.choices[0].finish_reason,
                               usage=answer.usage.model_dump() if answer.usage else None,content=answer.choices[0].message.content)
        if not data['nonstream']['content']: raise RuntimeError('Empty nonstream response')
        checkpoint(output_path,data); resource_gate()
        start=time.perf_counter(); first=None; pieces=[]; finish=None; chunks=0
        stream=client.chat.completions.create(**request,stream=True)
        try:
            for chunk in stream:
                chunks+=1
                if chunks>512: raise RuntimeError('SDK chunk budget exceeded')
                delta=chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    if first is None: first=time.perf_counter()-start
                    pieces.append(delta)
                if chunk.choices and chunk.choices[0].finish_reason: finish=chunk.choices[0].finish_reason
        finally: stream.close()
        data['sdk_stream']=dict(first_content_s=first,wall_s=time.perf_counter()-start,finish_reason=finish,
                                content=''.join(pieces),sdk_chunks=chunks)
        if first is None or finish is None: raise RuntimeError('Incomplete SDK stream')
        checkpoint(output_path,data); resource_gate()
        payload=dict(model=SERVED_NAME,messages=[dict(role='user',content=RAW_PROMPT)],max_tokens=16,temperature=0,
                     stream=True,stream_options=dict(include_usage=True))
        parser=ChatStream(); raw_chunks=[]; start=time.perf_counter()
        with httpx.stream('POST',base+'/chat/completions',json=payload,timeout=120,trust_env=False) as response:
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
        try: client.chat.completions.create(model='not-the-served-model',messages=[dict(role='user',content='hi')],max_tokens=1)
        except openai.APIStatusError as exc: negative['wrong_model']=dict(status=exc.status_code)
        long_text=' blue'*2300
        rendered=normalize_input_ids(tokenizer.apply_chat_template([dict(role='user',content=long_text)],tokenize=True,add_generation_prompt=True))
        if not 2048<len(rendered)<8192: raise RuntimeError('Bounded over-context fixture invalid')
        try: client.chat.completions.create(model=SERVED_NAME,messages=[dict(role='user',content=long_text)],max_tokens=16)
        except openai.APIStatusError as exc: negative['over_context']=dict(status=exc.status_code,rendered_prompt_tokens=len(rendered))
        if set(negative)!={'wrong_model','over_context'} or any(not 400<=r['status']<500 for r in negative.values()):
            raise RuntimeError('Expected 4xx contract not met')
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


def validate_artifact(data):
    import math
    def require(value,message):
        if not value: raise ValueError(message)
    def finite_number(value): return type(value) in (int,float) and math.isfinite(value)
    def usage_counts(usage):
        return isinstance(usage,dict) and all(type(usage.get(k)) is int and usage[k]>=0
            for k in ('prompt_tokens','completion_tokens','total_tokens')) and usage['completion_tokens']>0
    require(data.get('contract')==PROTOCOL and data.get('status')=='complete','incomplete/wrong protocol')
    require(data.get('revision')==REVISION and data.get('model_id')==MODEL_ID,'model identity')
    require(data.get('vllm')=='0.29.0' and all(isinstance(data.get(key),str) and data[key] for key in ('openai_sdk','httpx','transformers','torch','cuda','gpu')),'runtime pins/identity')
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
    require(len(resources)==2 and all(finite_number(r.get('available_ram_gib')) and r['available_ram_gib']>=24 and
                type(r.get('device_used_mib')) is int and 0<=r['device_used_mib']<=12288 for r in resources),'resource envelope')
    require(resources[-1]['device_used_mib']<=resources[0]['device_used_mib']+64,'device allocation recovered after teardown')
    require(set(data['negative'])=={'wrong_model','over_context'},'negative probes')
    require(all(type(row['status']) is int and 400<=row['status']<500 for row in data['negative'].values()),'negative 4xx')
    require(type(data['negative']['over_context']['rendered_prompt_tokens']) is int and
            2048<data['negative']['over_context']['rendered_prompt_tokens']<8192,'bounded token rejection')
    return True


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(description='CPU-only Topic 04 artifact contract validation')
    parser.add_argument('artifact',type=Path)
    args=parser.parse_args()
    validate_artifact(json.loads(args.artifact.read_text(encoding='utf-8')))
    print('Topic 04 complete artifact contract passed; not a quality or performance certification')
