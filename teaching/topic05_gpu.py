"""Provided bounded OFFLINE burst comparison; not a PagedAttention ablation.

Run only through the exact browser notebook after CPU prerequisites. Both engines
receive the same rendered strings; this measures cohort wall time, NOT TTFT,
individual client latency, open-loop capacity or free-account GPU availability.
Imports before explicit child execution are standard-library-only.
"""
import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import io
import zipfile
from typing import NotRequired, TypedDict

MODEL='Qwen/Qwen2.5-0.5B-Instruct'
REVISION='7ae557604adf67be50417f59c2c2f167def9a775'
PROTOCOL='topic05-offline-bursts-v1'
OUTPUT_POLICY='Exclude terminal configured EOS for rate; preserve actual native IDs. Pipeline native tensor includes prompt/padding; vLLM native IDs are completion-only.'
GENERATION_PROFILE={'mode':'greedy','repetition_penalty':1.0,'presence_penalty':0.0,
                    'frequency_penalty':0.0,'output_cap':16,'natural_eos':True,
                    'num_sequences':1,'seed':0}
SYSTEM='Classify the support ticket as billing, access, or bug. Output exactly one label and nothing else.'
CASES=[
 ('b1','I was charged twice for my monthly subscription. Please refund the duplicate.','billing'),
 ('b2','Please send an invoice for the payment we made last week.','billing'),
 ('b3','My credit card expired and the subscription payment was declined.','billing'),
 ('b4','I cancelled the subscription but a new charge appeared on my bank statement.','billing'),
 ('a1','I forgot my password and cannot sign in to my account.','access'),
 ('a2','Our new employee needs permission to open the team workspace.','access'),
 ('a3','I replaced my phone and no longer have my two-factor login codes.','access'),
 ('a4','My account is locked after too many failed sign-in attempts.','access'),
 ('g1','I can log in, but clicking Export crashes the application.','bug'),
 ('g2','After successful login, the Save button does nothing and my edits disappear.','bug'),
 ('g3','I am signed in. The search page shows an error for every search query.','bug'),
 ('g4','My account works, but every image I upload is displayed upside down.','bug'),
]


class Headroom(TypedDict):
    """Observed host/device resource sample; not an OS memory sandbox."""
    host_available_bytes: int
    gpu_free_mib: int
    gpu_used_mib: int


class GeneratedSequence(TypedDict):
    """Actual engine response and selected IDs; native API tensor scopes differ."""
    selected_ids: list[int]
    native_return_ids: list[int]
    finish_reason: str
    non_stop_id_count: int


class ComparisonReceipt(TypedDict):
    """Two owned child files and actual resource snapshots, not learner acceptance."""
    protocol: str
    model: str
    revision: str
    initial: Headroom
    children: dict[str, str]
    post: Headroom
    scope: str


class PromptRecord(TypedDict):
    """Exact rendered input, separately retained from native generated tensors."""
    case_id: str
    text: str
    input_ids: list[int]


class RequestRecord(TypedDict):
    """One actual prediction; the full attempted population remains in the score."""
    case_id: str
    cohort_size: int
    repeat: int
    output_ids: list[int]
    output: str
    gold: str
    native_return_ids: list[int]
    finish_reason: str
    non_stop_id_count: int
    correct: bool


class CohortRecord(TypedDict):
    """Synchronized library call interval, not an individual client latency."""
    repeat: int
    cohort_size: int
    case_ids: list[str]
    wall_s: float
    resource: Headroom


class BackendArtifact(TypedDict):
    """Literal source-bound child artifact; pending browser acceptance is separate."""
    protocol: str
    status: str
    backend: str
    model: str
    revision: str
    dtype: str
    prefix_caching: bool
    cohort_sizes: list[int]
    repeats: int
    output_cap: int
    versions: dict[str, str]
    device: str
    prompts: list[PromptRecord]
    cohorts: list[CohortRecord]
    requests: list[RequestRecord]
    initial: Headroom
    generation_profile: dict[str, object]
    output_id_policy: str
    source_sha256: str
    eos_ids: list[int]
    pad_token_id: int
    cache_block_size: NotRequired[int]
    kv_cache_memory_bytes: NotRequired[int]


class SummaryRecord(TypedDict):
    """One engine/size/repeat over all twelve attempts, not twelve client clocks."""
    engine: str
    cohort_size: int
    repeat: int
    attempted: int
    wall_s: float
    requests_per_s: float
    selected_non_stop_ids_per_s: float
    correct: int
    invalid: int


def load_pair(directory: Path) -> dict[str, BackendArtifact]:
    """Reload and validate actual files for viewers without running inference.

    Parameters
    ----------
    directory : Path
        Explicit current attempt directory containing both completed JSONs.

    Returns
    -------
    dict[str, BackendArtifact]
        Two literal validated artifacts with identical rendered input ledgers.

    Raises
    ------
    AssertionError
        Incomplete/corrupt identity, workload or cross-engine input mismatch.
    """
    pair={engine:json.loads((directory/(engine+'.json')).read_text())
          for engine in ('pipeline','vllm')}
    for engine,data in pair.items(): validate_backend(data,engine,[1,4],3)
    assert pair['pipeline']['prompts']==pair['vllm']['prompts']
    return pair


def summarize(pair: dict[str, BackendArtifact]) -> list[SummaryRecord]:
    """Reduce all attempts with a declared cohort clock; never drop bad outputs.

    Parameters
    ----------
    pair : dict[str, BackendArtifact]
        Literal validated artifacts, freshly reloaded by load_pair.

    Returns
    -------
    list[SummaryRecord]
        Twelve engine/size/repeat rows. Rate is12/sum(cohort wall seconds), not
        mean(1/request latency). Invalid labels remain in all12 attempts.
    """
    result=[]
    for engine,data in pair.items():
        validate_backend(data,engine,[1,4],3)
        for size in (1,4):
            for repeat in range(3):
                rows=[r for r in data['requests'] if r['cohort_size']==size and r['repeat']==repeat]
                wall=sum(c['wall_s'] for c in data['cohorts'] if c['cohort_size']==size and c['repeat']==repeat)
                result.append(dict(engine=engine,cohort_size=size,repeat=repeat,attempted=len(rows),
                    wall_s=wall,requests_per_s=len(rows)/wall,
                    selected_non_stop_ids_per_s=sum(r['non_stop_id_count'] for r in rows)/wall,
                    correct=sum(r['correct'] for r in rows),
                    invalid=sum(r['output'].strip().lower() not in ('billing','access','bug') for r in rows)))
    return result


def archive_attempt(directory: Path) -> str:
    """Return a bounded download link for only the explicit owned attempt files.

    Parameters
    ----------
    directory : Path
        Current named topic05-* folder. Partial attempts are preserved as partial;
        no status promotion, model cache, credentials or arbitrary globs.

    Returns
    -------
    str
        HTML data-download anchor plus exact inventory/digests. At most32MiB
        uncompressed; browser-download behaviour requires actual browser QA.

    Raises
    ------
    ValueError
        Bad attempt name, symlink, missing evidence or excessive file size.
    """
    if not directory.is_dir() or not directory.name.startswith('topic05-') or directory.is_symlink():
        raise ValueError('Explicit owned topic05-* attempt directory required')
    allowed=('pipeline.json','pipeline.log','pipeline.json.partial.tmp',
             'vllm.json','vllm.log','vllm.json.partial.tmp','index.json','worksheet.json')
    inventory=[];raw_files=[];total=0
    for name in allowed:
        path=directory/name
        if not path.exists(): continue
        if path.is_symlink() or not path.is_file(): raise ValueError('Reject symlink/non-file evidence')
        size=path.stat().st_size;total+=size
        if total>32*2**20: raise ValueError('Evidence exceeds32MiB; retain files individually')
        raw=path.read_bytes();raw_files.append((name,raw))
        inventory.append(dict(name=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    if not raw_files: raise ValueError('No actual evidence files; nothing to download')
    memory=io.BytesIO()
    with zipfile.ZipFile(memory,'w',compression=zipfile.ZIP_DEFLATED) as bundle:
        for name,raw in raw_files: bundle.writestr(name,raw)
        bundle.writestr('inventory.json',json.dumps(inventory,indent=2)+'\n')
    raw=memory.getvalue();encoded=base64.b64encode(raw).decode('ascii')
    # Validated name is additionally reduced to safe ASCII; no path interpolation.
    name=''.join(c for c in directory.name if c.isascii() and (c.isalnum() or c=='-'))
    return '<a download="'+name+'.zip" href="data:application/zip;base64,'+encoded+'">Download this actual attempt ZIP</a><p>ZIP SHA256 '+hashlib.sha256(raw).hexdigest()+'; '+str(len(raw_files))+' actual files; partial attempts stay partial.</p>'


def terminate_owned_group(process: subprocess.Popen) -> None:
    """Bound cleanup of this exact child session, including exceptional exits.

    Parameters
    ----------
    process : subprocess.Popen
        The process this supervisor created with start_new_session=True. Never
        an arbitrary PID. Cleanup fits below the bridge's ten-second TERM grace.

    Returns
    -------
    None
        Parent reaped and owned process group no longer present.

    Raises
    ------
    RuntimeError
        The owned group still exists after bounded TERM/KILL cleanup. This is
        not accepted teardown; preserve logs and disconnect the disposable VM.
    """
    try:
        os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:
        process.wait(timeout=2)
        return
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid,signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=2)
    for _ in range(20):
        try:
            os.killpg(process.pid,0)
        except ProcessLookupError:
            return
        time.sleep(.1)
    raise RuntimeError('Owned child group remains after cleanup; teardown NOT accepted')


def headroom(startup: bool = False) -> Headroom:
    """Check provisional small-model safety bounds without allocating GPU tensors.

    Parameters
    ----------
    startup : bool, default False
        Before load require idle GPU and6GiB free device/host. During runs require
        2GiB free host and512MiB free device. These NEW bounds are provisional,
        not transferred Topic04 acceptance or weakened operator/judge gates.

    Returns
    -------
    Headroom
        Actual observed available bytes and MiB on device0.

    Raises
    ------
    RuntimeError
        Missing CUDA diagnostics or insufficient/occupied startup resources.
    """
    fields={line.split(':')[0]:int(line.split()[1])*1024
            for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')}
    raw=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.free,memory.used',
        '--format=csv,noheader,nounits'],text=True).strip()
    free,used=map(int,raw.split(','))
    record=Headroom(host_available_bytes=fields['MemAvailable'],gpu_free_mib=free,gpu_used_mib=used)
    if record['host_available_bytes']<(6 if startup else 2)*2**30 or free<(6144 if startup else 512):
        raise RuntimeError('Stop: insufficient provisional headroom '+repr(record))
    if startup and used>256: raise RuntimeError('Stop: use an otherwise idle owned GPU '+repr(record))
    return record


def run_comparison(directory: Path, cohort_sizes: list[int], repeats: int = 3) -> ComparisonReceipt:
    """Run two isolated sequential owned child processes; retain all partial attempts.

    Parameters
    ----------
    directory : Path
        New exclusive evidence directory. Never an existing run or model cache.
    cohort_sizes : list[int]
        Exactly[1,4] required offline burst sizes, not HTTP clients. This changes
        submitted cohort size; each engine sees the same12cases/repeat.
    repeats : int, default 3
        Exactly3 repeated population passes per size and engine. No p95 claim.

    Returns
    -------
    ComparisonReceipt
        Protocol, two child artifact filenames, idle post-device sample and scope.
        Raw outputs remain in child files, never fabricated after a failure.

    Raises
    ------
    ValueError
        Wrong workload or pre-existing output directory.
    RuntimeError
        Resource, child, timeout or cleanup failure; retain logs/partial JSON.
    """
    if cohort_sizes!=[1,4] or any(type(n) is not int for n in cohort_sizes) or type(repeats) is not int or repeats!=3:
        raise ValueError('Required integer sizes[1,4], repeats3')
    directory.mkdir(parents=True,exist_ok=False)
    initial=headroom(True)
    receipts={}
    first_prompts=None
    def interrupted(signum: int, frame: object) -> None:
        """Convert worker TERM into an exception so owned-child finally runs.

        Parameters
        ----------
        signum : int
            SIGTERM delivered by the owned bridge.
        frame : object
            Python signal frame, unused.

        Returns
        -------
        None
            Always raises InterruptedError; no new engine starts afterward.
        """
        raise InterruptedError('Owned supervisor interrupted by signal '+str(signum))

    previous_term=signal.signal(signal.SIGTERM,interrupted)
    try:
      for backend in ['pipeline','vllm']:
        output=directory/(backend+'.json')
        log=directory/(backend+'.log')
        command=[sys.executable,'-I',str(Path(__file__).resolve()),'--child',backend,
            '--output',str(output.resolve()),'--cohort-sizes',*map(str,cohort_sizes),'--repeats',str(repeats)]
        environment=dict(os.environ,VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_USAGE_STATS_DO_NOT_TRACK='1',
            DO_NOT_TRACK='1',TOKENIZERS_PARALLELISM='false')
        with log.open('xb') as stream:
            process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,
                env=environment,start_new_session=True)
            try:
                code=process.wait(timeout=420)
            except subprocess.TimeoutExpired:
                raise RuntimeError('Timeout: partial '+backend+' evidence/log preserved')
            finally:
                # Includes KeyboardInterrupt, bridge TERM, timeout and read failures.
                terminate_owned_group(process)
        if code!=0: raise RuntimeError('Child failed: '+backend+'; retain its log/partial JSON')
        data=json.loads(output.read_text())
        validate_backend(data,backend,cohort_sizes,repeats)
        if first_prompts is None: first_prompts=data['prompts']
        else: assert data['prompts']==first_prompts,'Cross-engine rendered input ledger mismatch'
        headroom(True)
        receipts[backend]=output.name
    finally:
        signal.signal(signal.SIGTERM,previous_term)
    result=dict(protocol=PROTOCOL,model=MODEL,revision=REVISION,initial=initial,
        children=receipts,post=headroom(True),scope='offline cohort wall comparison; no TTFT, client latency, mechanism ablation, learner or judge acceptance')
    (directory/'index.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def validate_backend(data: BackendArtifact, backend: str,
                     sizes: list[int], repeats: int) -> None:
    """Reject incomplete or wrongly labelled artifacts; does not award browser acceptance.

    Parameters
    ----------
    data : BackendArtifact
        Literal child result, with runtime/ledger/cohorts/request rows.
    backend : str
        Required engine label, pipeline or vllm.
    sizes : list[int]
        Required[1,4] cohort sizes.
    repeats : int
        Required3 repeated full populations.

    Returns
    -------
    None
        Checks identity, finite clocks, complete unique attempts and all-attempt
        score consistency. This is artifact integrity, not model quality.
    """
    assert backend in ('pipeline','vllm') and sizes==[1,4] and all(type(n) is int for n in sizes)
    assert type(repeats) is int and repeats==3
    assert data['protocol']==PROTOCOL and data['status']=='complete' and data['backend']==backend
    assert data['model']==MODEL and data['revision']==REVISION
    assert data['cohort_sizes']==sizes and all(type(n) is int for n in data['cohort_sizes'])
    assert type(data['repeats']) is int and data['repeats']==repeats
    assert type(data['output_cap']) is int and data['output_cap']==16
    assert data['source_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert data['dtype']=='float16' and data['prefix_caching'] is False
    assert data['generation_profile']==GENERATION_PROFILE and data['output_id_policy']==OUTPUT_POLICY
    assert data['versions']['python']=='3.11.17'
    assert data['versions']['transformers']=='5.17.0' and data['versions']['torch']=='2.13.0+cu129'
    assert isinstance(data['device'],str) and data['device'].strip()
    if backend=='vllm':
        assert data['versions']['vllm_distribution']=='0.30.0+cu129' and data['versions']['vllm_module']=='0.30.0'
        assert type(data['cache_block_size']) is int and data['cache_block_size']>0
        assert data['kv_cache_memory_bytes']==256*2**20
    def ids(values: list[int]) -> None:
        """Reject malformed native IDs without importing a tokenizer.

        Parameters
        ----------
        values : list[int]
            Literal IDs. Booleans, strings and negative IDs are rejected.

        Returns
        -------
        None
            Validates structure only, not model vocabulary or decoded text.
        """
        assert isinstance(values,list) and all(type(n) is int and n>=0 for n in values)
    ids(data['eos_ids']);assert sorted(set(data['eos_ids']))==[151643,151645]
    assert data['pad_token_id']==151645 and type(data['pad_token_id']) is int
    prompts=data['prompts'];assert isinstance(prompts,list) and len(prompts)==len(CASES)
    assert [p['case_id'] for p in prompts]==[c[0] for c in CASES]
    for p,case in zip(prompts,CASES,strict=True):
        ids(p['input_ids']);assert p['input_ids'] and len(p['input_ids'])+16<=512
        assert isinstance(p['text'],str) and SYSTEM in p['text'] and case[1] in p['text']
    ledger={p['case_id']:p for p in prompts}
    rows=data['requests']; expected={(c[0],s,r) for c in CASES for s in sizes for r in range(repeats)}
    assert isinstance(rows,list) and len(rows)==len(expected) and {(x['case_id'],x['cohort_size'],x['repeat']) for x in rows}==expected
    for row in rows:
        assert type(row['cohort_size']) is int and type(row['repeat']) is int
        ids(row['output_ids']);ids(row['native_return_ids']);assert len(row['output_ids'])<=16
        assert not any(n in data['eos_ids'] for n in row['output_ids'][:-1]),'Selected IDs cannot continue after an EOS'
        assert isinstance(row['output'],str) and type(row['correct']) is bool
        assert row['finish_reason'] in ('stop','length')
        assert type(row['non_stop_id_count']) is int
        assert row['non_stop_id_count']==sum(n not in data['eos_ids'] for n in row['output_ids'])
        if row['finish_reason']=='length': assert len(row['output_ids'])==16 and not any(n in data['eos_ids'] for n in row['output_ids'])
        if backend=='vllm': assert row['native_return_ids']==row['output_ids']
        else:
            native=row['native_return_ids'];prompt=ledger[row['case_id']]['input_ids'];padding=0
            while native[padding:padding+len(prompt)]!=prompt:
                assert padding<len(native) and native[padding]==data['pad_token_id'];padding+=1
            tail=native[padding+len(prompt):];stop=next((j for j,n in enumerate(tail) if n in data['eos_ids']),None)
            selected=tail if stop is None else tail[:stop+1]
            assert selected==row['output_ids'] and len(tail)<=16
            assert row['finish_reason']==('length' if stop is None else 'stop')
        gold=next(c[2] for c in CASES if c[0]==row['case_id'])
        assert row['gold']==gold and row['correct']==(row['output'].strip().lower()==gold)
    expected_cohorts=[(r,s,tuple(c[0] for c in CASES[first:first+s])) for r in range(repeats)
                     for s in (sizes if r%2==0 else list(reversed(sizes))) for first in range(0,len(CASES),s)]
    cohorts=data['cohorts'];assert isinstance(cohorts,list) and len(cohorts)==len(expected_cohorts)
    assert [(c['repeat'],c['cohort_size'],tuple(c['case_ids'])) for c in cohorts]==expected_cohorts
    for record in cohorts:
        assert type(record['repeat']) is int and type(record['cohort_size']) is int
        assert type(record['wall_s']) in (int,float) and math.isfinite(record['wall_s']) and record['wall_s']>0
        resource=record['resource']
        assert all(type(resource[k]) is int and resource[k]>=0 for k in Headroom.__annotations__)
        assert resource['host_available_bytes']>=2*2**30 and resource['gpu_free_mib']>=512
    initial=data['initial'];assert all(type(initial[k]) is int and initial[k]>=0 for k in Headroom.__annotations__)
    assert initial['host_available_bytes']>=6*2**30 and initial['gpu_free_mib']>=6144 and initial['gpu_used_mib']<=256


def child(backend: str, output: Path, sizes: list[int], repeats: int) -> None:
    """Execute an actual child ONLY after the notebook's explicit GPU opt-in.

    Parameters
    ----------
    backend : str
        pipeline or vllm. GPU packages are imported here, not by CPU helper import.
    output : Path
        Exclusive result path; checkpoint after each cohort, retain failures.
    sizes : list[int]
        Required cohort sizes[1,4].
    repeats : int
        Required3 passes, interleaved by repeat then size.

    Returns
    -------
    None
        Writes actual raw token/output/clock records. No teacher solution code.
    """
    if sizes!=[1,4] or any(type(n) is not int for n in sizes) or type(repeats) is not int or repeats!=3 or output.exists():
        raise ValueError('Wrong/existing attempt')
    resource=headroom(True)
    import torch
    from importlib.metadata import version
    from transformers import AutoTokenizer, GenerationConfig
    assert version('transformers')=='5.17.0'
    assert version('torch')=='2.13.0+cu129' and sys.version.split()[0]=='3.11.17'
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION,trust_remote_code=False)
    tokenizer.padding_side='left'
    tokenizer.pad_token=tokenizer.eos_token
    eos_ids=set(tokenizer.encode(tokenizer.eos_token,add_special_tokens=False))
    generation_config=GenerationConfig.from_pretrained(MODEL,revision=REVISION)
    configured_eos=generation_config.eos_token_id
    eos_ids.update(configured_eos if isinstance(configured_eos,list) else [configured_eos])
    prompts=[tokenizer.apply_chat_template([{'role':'system','content':SYSTEM},{'role':'user','content':text}],
        tokenize=False,add_generation_prompt=True) for _,text,_ in CASES]
    ledger=[dict(case_id=case[0],text=prompt,input_ids=tokenizer(prompt,add_special_tokens=False)['input_ids'])
            for case,prompt in zip(CASES,prompts)]
    assert all(len(p['input_ids'])+16<=512 for p in ledger)
    identity=dict(protocol=PROTOCOL,status='partial',backend=backend,model=MODEL,revision=REVISION,
        dtype='float16',prefix_caching=False,cohort_sizes=sizes,repeats=repeats,output_cap=16,
        versions={'python':sys.version.split()[0],'transformers':version('transformers'),'torch':version('torch')},
        device=torch.cuda.get_device_name(0),prompts=ledger,cohorts=[],requests=[],initial=resource)
    identity['generation_profile']=dict(GENERATION_PROFILE)
    identity['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    identity['pad_token_id']=tokenizer.pad_token_id
    def save() -> None:
        """Checkpoint actual child evidence without replacing a different attempt.

        Returns
        -------
        None
            Rewrites only this child's explicitly owned partial result file.
        """
        temporary=output.with_suffix('.json.partial.tmp')
        with temporary.open('w',encoding='utf8') as stream:
            stream.write(json.dumps(identity,ensure_ascii=False,indent=2)+'\n')
            stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,output) # Atomic within this exact owned attempt.
    save()
    if backend=='pipeline':
        from transformers import AutoModelForCausalLM, pipeline
        # Single-GPU placement uses ordinary PyTorch .to(), not device_map:
        # the frozen serving profile deliberately contains no Accelerate.
        model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,dtype=torch.float16,
            trust_remote_code=False,attn_implementation='sdpa').to('cuda:0')
        assert model.device.type=='cuda' and model.dtype==torch.float16
        engine=pipeline('text-generation',model=model,tokenizer=tokenizer)
        configured_eos=model.generation_config.eos_token_id
        eos_ids.update(configured_eos if isinstance(configured_eos,list) else [configured_eos])
    else:
        import vllm
        from vllm import LLM, SamplingParams
        assert version('vllm')=='0.30.0+cu129' and vllm.__version__=='0.30.0'
        identity['versions'].update(vllm_distribution=version('vllm'),vllm_module=vllm.__version__)
        engine=LLM(model=MODEL,revision=REVISION,tokenizer_revision=REVISION,dtype='float16',
            max_model_len=512,max_num_seqs=4,max_num_batched_tokens=256,
            kv_cache_memory_bytes=256*2**20,enable_prefix_caching=False,
            enable_chunked_prefill=True,enforce_eager=True,trust_remote_code=False)
        parameters=SamplingParams(temperature=0,max_tokens=16,seed=0,repetition_penalty=1.0,
            presence_penalty=0.0,frequency_penalty=0.0)
        config=engine.llm_engine.vllm_config
        assert config.cache_config.enable_prefix_caching is False
        assert config.scheduler_config.max_num_seqs==4
        assert config.scheduler_config.max_num_batched_tokens==256
        assert config.model_config.max_model_len==512
        identity['cache_block_size']=config.cache_config.block_size
        identity['kv_cache_memory_bytes']=config.cache_config.kv_cache_memory_bytes
    eos_ids.discard(None)
    identity['eos_ids']=sorted(eos_ids)
    identity['output_id_policy']=OUTPUT_POLICY
    def generate(indices: list[int]) -> tuple[list[GeneratedSequence], float]:
        """Time one actual offline cohort; retain every generated-ID list.

        Parameters
        ----------
        indices : list[int]
            Case indices of one nonempty burst cohort in declared order.

        Returns
        -------
        tuple[list[GeneratedSequence], float]
            Actual selected/native IDs, stopping reason and non-stop count per
            case, plus synchronized library wall seconds. Native API scopes
            differ; do not compare full pipeline tensors with vLLM output IDs.
            No individual client clocks or first-token arrival are observable here.
        """
        inputs=[prompts[i] for i in indices]
        headroom()
        torch.cuda.synchronize();start=time.perf_counter()
        if backend=='pipeline':
            raw=engine(inputs,batch_size=len(indices),max_new_tokens=16,do_sample=False,
                repetition_penalty=1.0,num_beams=1,num_return_sequences=1,
                return_tensors=True,pad_token_id=tokenizer.pad_token_id,add_special_tokens=False)
        else: raw=engine.generate(inputs,parameters,use_tqdm=False)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        outputs=[]
        for i,record in zip(indices,raw):
            if backend=='pipeline':
                ids=record[0]['generated_token_ids']
                if hasattr(ids,'tolist'): ids=ids.tolist()
                native=list(ids)
                expected=ledger[i]['input_ids']
                padding=0
                while ids[padding:padding+len(expected)]!=expected:
                    assert padding<len(ids) and ids[padding]==tokenizer.pad_token_id,'Unexpected pipeline input prefix'
                    padding+=1
                ids=ids[padding+len(expected):]
                stop=next((j for j,n in enumerate(ids) if n in eos_ids),None)
                if stop is not None: ids=ids[:stop+1]
                finish='stop' if stop is not None else 'length'
            else:
                assert list(record.prompt_token_ids)==ledger[i]['input_ids'],'Wrong vLLM rendered input IDs'
                ids=list(record.outputs[0].token_ids)
                native=list(ids);finish=record.outputs[0].finish_reason
            outputs.append(GeneratedSequence(selected_ids=[int(n) for n in ids],
                native_return_ids=[int(n) for n in native],finish_reason=finish,
                non_stop_id_count=sum(n not in eos_ids for n in ids)))
        assert len(outputs)==len(indices)
        return outputs,elapsed
    # Warmup outside each clock/population; the actual paths remain observable.
    for size in sizes: generate(list(range(size)))
    for repeat in range(repeats):
        for size in (sizes if repeat%2==0 else list(reversed(sizes))):
            for first in range(0,len(CASES),size):
                indices=list(range(first,first+size))
                sequences,elapsed=generate(indices)
                identity['cohorts'].append(dict(repeat=repeat,cohort_size=size,
                    case_ids=[CASES[i][0] for i in indices],wall_s=elapsed,
                    resource=headroom()))
                for i,sequence in zip(indices,sequences):
                    tokens=sequence['selected_ids']
                    text=tokenizer.decode(tokens,skip_special_tokens=True)
                    identity['requests'].append(dict(case_id=CASES[i][0],cohort_size=size,
                        repeat=repeat,output_ids=tokens,output=text,gold=CASES[i][2],
                        native_return_ids=sequence['native_return_ids'],finish_reason=sequence['finish_reason'],
                        non_stop_id_count=sequence['non_stop_id_count'],
                        correct=text.strip().lower()==CASES[i][2]))
                save()
    identity['status']='complete';validate_backend(identity,backend,sizes,repeats);save()
    # GPU process termination is supervised by the parent; no in-process reuse.


def main() -> None:
    """Explicit child CLI; never run as a CPU import side effect.

    Returns
    -------
    None
        Executes only with the supplied --child argument and owned output path.
    """
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child',choices=['pipeline','vllm'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cohort-sizes',type=int,nargs='+',required=True)
    parser.add_argument('--repeats',type=int,required=True)
    a=parser.parse_args();child(a.child,a.output,a.cohort_sizes,a.repeats)


if __name__=='__main__': main()
