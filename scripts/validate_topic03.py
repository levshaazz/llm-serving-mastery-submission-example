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


def run_lab(snapshot,output_path):
    import torch,transformers,bitsandbytes
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
    from transformers.generation.streamers import BaseStreamer
    if transformers.__version__!='4.51.3' or bitsandbytes.__version__!='0.50.2':
        raise RuntimeError('Pinned Transformers/bitsandbytes versions required')
    output_path=Path(output_path)
    if output_path.exists(): raise FileExistsError('Preserve prior evidence: choose a fresh output path')
    output_path.parent.mkdir(parents=True,exist_ok=True)
    preflight=resource_gate(require_idle=True)
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required; recorded replay is not execution evidence')
    tokenizer=AutoTokenizer.from_pretrained(str(snapshot),local_files_only=True)
    class TokenClock(BaseStreamer):
        def __init__(self): self.seen_prompt=False; self.first=None
        def put(self,value):
            if not self.seen_prompt: self.seen_prompt=True; return
            # Same callback overhead in both treatments; not HTTP first-content time.
            torch.cuda.synchronize()
            if self.first is None: self.first=time.perf_counter()
        def end(self): pass
    def one(model,case,force):
        messages=[dict(role='user',content=case['text'])]
        rendered=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_tensors='pt').to('cuda')
        kwargs=dict(do_sample=False,max_new_tokens=MAX_NEW,pad_token_id=tokenizer.eos_token_id)
        if force: kwargs['min_new_tokens']=MAX_NEW
        clock=TokenClock(); torch.cuda.synchronize(); begin=time.perf_counter()
        with torch.inference_mode(): result=model.generate(ids,streamer=clock,**kwargs)
        torch.cuda.synchronize(); end=time.perf_counter()
        selected=result[0,ids.shape[-1]:].tolist()
        input_ids=ids[0].tolist()
        if clock.first is None or not selected: raise RuntimeError('No generated token callback')
        row=dict(prompt_tokens=len(input_ids),input_token_ids=input_ids,rendered_chat=rendered,
                 output_tokens=len(selected),output_token_ids=selected,
                 output=tokenizer.decode(selected,skip_special_tokens=True),
                 output_token_pieces=tokenizer.convert_ids_to_tokens(selected),
                 stop_reason='fixed_length' if force else ('eos' if selected[-1] in eos_ids else 'length'),
                 ttft_s=clock.first-begin,wall_s=end-begin,
                 post_first_s_per_token=(end-clock.first)/(len(selected)-1) if len(selected)>1 else None)
        del result,ids
        return row
    data=dict(contract=PROTOCOL,status='running',utc_started=datetime.now(timezone.utc).isoformat(),
              model_id=MODEL_ID,revision=REVISION,gpu=torch.cuda.get_device_name(0),
              torch=torch.__version__,cuda=torch.version.cuda,transformers=transformers.__version__,
              bitsandbytes=bitsandbytes.__version__,trials=TRIALS,max_new_tokens=MAX_NEW,
              quality_mode='natural_eos',timing_mode='fixed_output_length',formats={},resource_samples=[preflight],
              clock='synchronized generate wall; first generated-ID streamer callback; post-first proxy includes return overhead, not last-emission interval; not HTTP or pure GPU time',
              quantization=dict(quant_type='nf4',compute_dtype='float16',double_quant=False),
              limitations=['four-prompt smoke, not quality benchmark','fixed format order, three trials, uncontrolled timing variance',
                           'allocated/reserved are PyTorch allocator counters, not complete process/device memory'])
    checkpoint(output_path,data)
    model=None
    try:
        for label in ('fp16','nf4_w4a16'):
            resource_gate(); gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            before=torch.cuda.memory_allocated()
            quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.float16,bnb_4bit_use_double_quant=False) if label=='nf4_w4a16' else None
            begin=time.perf_counter()
            model=AutoModelForCausalLM.from_pretrained(str(snapshot),local_files_only=True,torch_dtype=torch.float16,
                    quantization_config=quant,device_map={'':0}).eval()
            torch.cuda.synchronize()
            eos=model.generation_config.eos_token_id
            eos_ids=eos if isinstance(eos,list) else [eos]
            entry=dict(load_s=time.perf_counter()-begin,rows=[],status='running',baseline_allocated_bytes=before,
                       eos_token_ids=eos_ids,quantized_linear_modules=sum(type(m).__name__=='Linear4bit' for m in model.modules()))
            data['formats'][label]=entry; checkpoint(output_path,data)
            _=one(model,PROMPTS[0],True)  # Exactly one warmup per format, excluded.
            for case in PROMPTS:
                row=dict(bucket=case['bucket'],prompt=case['text'],quality_sample=one(model,case,False),timing_trials=[])
                entry['rows'].append(row); checkpoint(output_path,data)
                for trial in range(TRIALS):
                    resource_gate(); row['timing_trials'].append(one(model,case,True)); checkpoint(output_path,data)
                for field in ('ttft_s','wall_s'):
                    row['p95_'+field]=max(t[field] for t in row['timing_trials'])
                    row['median_'+field]=statistics.median(t[field] for t in row['timing_trials'])
                checkpoint(output_path,data)
            entry.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),status='complete')
            del model; model=None; gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
            entry['post_cleanup_allocated_bytes']=torch.cuda.memory_allocated()
            if entry['post_cleanup_allocated_bytes']>before+64*2**20: raise RuntimeError('STOP: live allocation did not recover')
            entry['cleanup_verified']=True
            data['resource_samples'].append(resource_gate()); checkpoint(output_path,data)
        data['status']='complete'
    except BaseException as exc:
        data['status']='failed'; data['error_type']=type(exc).__name__; raise
    finally:
        if model is not None: del model
        gc.collect(); torch.cuda.empty_cache()
        data['utc_finished']=datetime.now(timezone.utc).isoformat(); checkpoint(output_path,data)
    return data


def validate_artifact(data):
    def require(value,message):
        if not value: raise ValueError(message)
    require(data.get('contract')==PROTOCOL and data.get('status')=='complete','incomplete or wrong protocol')
    require(data.get('revision')==REVISION and data.get('model_id')==MODEL_ID,'model identity')
    require(data.get('transformers')=='4.51.3' and data.get('bitsandbytes')=='0.50.2','runtime pins')
    require(all(isinstance(data.get(key),str) and data[key] for key in ('torch','cuda','gpu')),'runtime identity')
    require(data.get('quantization')==dict(quant_type='nf4',compute_dtype='float16',double_quant=False),'quantization config')
    require(data.get('trials')==TRIALS and data.get('max_new_tokens')==MAX_NEW,'trial budget')
    require(data.get('quality_mode')=='natural_eos' and data.get('timing_mode')=='fixed_output_length','separate quality/timing')
    require(set(data.get('formats',{}))=={'fp16','nf4_w4a16'},'paired formats')
    paired=[]
    for label in ('fp16','nf4_w4a16'):
        entry=data['formats'][label]
        require(entry.get('status')=='complete' and len(entry.get('rows',[]))==4,'complete format rows')
        require(entry.get('quantized_linear_modules',-1)==0 if label=='fp16' else entry.get('quantized_linear_modules',0)>0,'actual module treatment')
        require(all(type(entry.get(key)) is int and entry[key]>=0 for key in ('peak_allocated_bytes','peak_reserved_bytes','baseline_allocated_bytes','post_cleanup_allocated_bytes')),'finite integer byte counters')
        require(0<entry['peak_allocated_bytes']<=entry['peak_reserved_bytes'],'allocator counters')
        require(entry.get('cleanup_verified') is True and entry['post_cleanup_allocated_bytes']<=entry['baseline_allocated_bytes']+64*2**20,'cleanup verified')
        require(math.isfinite(entry['load_s']) and entry['load_s']>0,'load clock')
        inputs=[]
        for case,row in zip(PROMPTS,entry['rows']):
            require(row['bucket']==case['bucket'] and row['prompt']==case['text'],'paired literal prompt')
            require(len(row['timing_trials'])==TRIALS,'three raw timing trials')
            samples=[row['quality_sample']]+row['timing_trials']
            for index,sample in enumerate(samples):
                require(sample['prompt_tokens']==len(sample['input_token_ids'])>0,'prompt IDs/count')
                require(sample['output_tokens']==len(sample['output_token_ids'])==len(sample['output_token_pieces'])>0,'output IDs/count')
                require(all(type(x) is int and x>=0 for x in sample['input_token_ids']+sample['output_token_ids']),'token ID types')
                require(sample['input_token_ids']==samples[0]['input_token_ids'] and sample['rendered_chat']==samples[0]['rendered_chat'],'within-prompt identity')
                require(math.isfinite(sample['wall_s']) and math.isfinite(sample['ttft_s']) and 0<sample['ttft_s']<=sample['wall_s'],'finite ordered clocks')
                expected=(sample['wall_s']-sample['ttft_s'])/(sample['output_tokens']-1) if sample['output_tokens']>1 else None
                require(sample['post_first_s_per_token']==expected,'post-first denominator')
                require(sample['output_tokens']<=MAX_NEW,'bounded output')
                if index: require(sample['output_tokens']==MAX_NEW and sample['stop_reason']=='fixed_length','fixed timing length')
                else:
                    is_eos=sample['output_token_ids'][-1] in entry['eos_token_ids']
                    require(sample['stop_reason']==('eos' if is_eos else 'length'),'natural stop reason')
                    require(is_eos or sample['output_tokens']==MAX_NEW,'natural termination requires EOS or cap')
            for field in ('ttft_s','wall_s'):
                values=[sample[field] for sample in row['timing_trials']]
                require(row['p95_'+field]==max(values) and row['median_'+field]==statistics.median(values),'raw summary arithmetic')
            inputs.append((samples[0]['input_token_ids'],samples[0]['rendered_chat']))
        paired.append(inputs)
    require(paired[0]==paired[1],'identical tokenized inputs across formats')
    return True


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(description='CPU-only Topic 03 artifact contract validation')
    parser.add_argument('artifact',type=Path)
    args=parser.parse_args()
    validate_artifact(json.loads(args.artifact.read_text(encoding='utf-8')))
    print('Topic 03 complete artifact contract passed; not a quality or performance certification')
