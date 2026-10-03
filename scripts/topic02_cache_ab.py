"""Bounded Colab KV-cache A/B. Optional teaching evidence, not the required lab.

Run in a fresh subprocess with an outer 600-second timeout. No private course
source, credentials, Drive mount or student data is needed. Imports are CPU-only.
"""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time

PROTOCOL = 'topic02-cache-ab-colab-v1'
MODEL = 'Qwen/Qwen2.5-1.5B-Instruct'
REVISION = '989aa7980e4cf806f80c7fef2b1adb7bc71aa306'
PROMPT = [151644,8948,198,2610,525,264,10950,17847,13,17841,448,1172,279,11223,5109,13,151645,198,151644,872,198,23526,279,8500,448,2326,5109,25,220,17,11,220,19,11,220,21,11,151645,198,151644,77091,198]
CONTINUATION = [23,11,220,16,15,11,220,16,17,151645]
PAIRS = 5
GIB = 1024**3
ATOL, RTOL = .05, .01


def plan(cached):
    """Replay the recorded path. Selected EOS is never fed back into the model."""
    return [(0 if not cached or i == 0 else len(PROMPT)+i-1, len(PROMPT)+i)
            for i in range(len(CONTINUATION))]


def summarize(trials):
    if len(trials) != PAIRS*2:
        raise ValueError('Need all five paired trials')
    for pair in range(PAIRS):
        rows = [r for r in trials if r['pair'] == pair]
        expected = ['no_cache','cache'] if pair % 2 == 0 else ['cache','no_cache']
        if [r['mode'] for r in rows] != expected:
            raise ValueError('Paired alternating order changed')
    for r in trials:
        if not all(math.isfinite(r[k]) and r[k] > 0 for k in ['wall_ms','cuda_interval_ms']):
            raise ValueError('Invalid interval')
        if not r['logits_allclose'] or not r['argmax_equal']:
            raise ValueError('Cannot claim speedup for divergent outputs')
    medians = {mode: statistics.median(r['wall_ms'] for r in trials if r['mode']==mode)
               for mode in ['no_cache','cache']}
    paired = [next(r['wall_ms'] for r in trials if r['pair']==i and r['mode']=='no_cache') /
              next(r['wall_ms'] for r in trials if r['pair']==i and r['mode']=='cache') for i in range(PAIRS)]
    return {'median_wall_ms':medians,'ratio_of_medians':medians['no_cache']/medians['cache'],
            'paired_wall_ratios':paired,'median_paired_wall_ratio':statistics.median(paired)}


def run(output, allow_download):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    if transformers.__version__ != '4.51.3' or not torch.cuda.is_available():
        raise RuntimeError('Require pinned Transformers 4.51.3 and CUDA')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    def stop(signum, frame):
        raise TimeoutError('Bounded experiment interrupted')
    signal.signal(signal.SIGALRM, stop)
    signal.signal(signal.SIGTERM, stop)
    signal.alarm(570)
    report = {'protocol':PROTOCOL,'status':'started',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'model':{'id':MODEL,'revision':REVISION},
        'input':{'prompt_ids':PROMPT,'fixed_continuation_ids':CONTINUATION,
                 'prompt_text':'Continue the sequence with three numbers: 2, 4, 6,',
                 'system_text':'You are a helpful assistant. Reply with only the requested numbers.'},
        'contract':{'pairs':PAIRS,'warmup_paths_per_mode':1,'dtype':'float16',
            'attention':'sdpa','logits_to_keep':1,'atol':ATOL,'rtol':RTOL,
            'input_policy':'teacher forcing of the same recorded IDs; no free generation',
            'timing':'one prefill plus nine following forwards; synchronized wall and CUDA stream interval; GPU output selection in both paths',
            'excluded':'load, download, tokenization, GPU-to-CPU copies, validation, profiler and resource checks',
            'limits':['five pairs on one runtime are not a latency distribution or production capacity',
                'CUDA interval is not summed kernel execution time; no HBM counters measured',
                'cache changes forward shapes; SDPA may select different kernels',
                'Colab runtime is managed; no claim of the WSL cgroup sandbox',
                'Hugging Face cache persists in this runtime, not across Colab runtime deletion'],
            'hard_timeout_seconds':600,'allocator_cap_gib':12,'minimum_host_available_gib':24},
        'environment':{'torch':torch.__version__,'transformers':transformers.__version__,
            'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
            'driver':subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip()},
        'trials':[],'resources':[]}
    model = tokenizer = inputs = reference = None
    def save():
        (output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    def gate(stage):
        mem = {line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')}
        free,total = torch.cuda.mem_get_info()
        processes = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip().splitlines()
        if any(p.strip().isdigit() and int(p) != os.getpid() for p in processes):
            raise RuntimeError('STOP: unrelated GPU compute process')
        state = {'stage':stage,'host_available_bytes':mem['MemAvailable'],'device_used_bytes':total-free,
                 'allocated_bytes':torch.cuda.memory_allocated(),'reserved_bytes':torch.cuda.memory_reserved()}
        report['resources'].append(state)
        if mem['MemAvailable'] < 24*GIB or total-free > 12*GIB:
            raise RuntimeError('STOP: resource envelope exceeded')
    try:
        gate('before_load')
        torch.cuda.set_per_process_memory_fraction(min(.8,12*GIB/torch.cuda.get_device_properties(0).total_memory))
        cache = os.environ.get('LSM_MODEL_CACHE','/content/lsm-model-cache')
        try:
            model_path = snapshot_download(MODEL,revision=REVISION,cache_dir=cache,local_files_only=True)
            report['downloaded_missing_snapshot'] = False
        except LocalEntryNotFoundError:
            if not allow_download:
                raise RuntimeError('Pinned snapshot missing; explicit --allow-download required')
            model_path = snapshot_download(MODEL,revision=REVISION,cache_dir=cache,
                allow_patterns=['*.json','*.safetensors','*.txt','*.model'],max_workers=2)
            report['downloaded_missing_snapshot'] = True
        tokenizer = AutoTokenizer.from_pretrained(model_path,local_files_only=True)
        actual = tokenizer.apply_chat_template([
            {'role':'system','content':report['input']['system_text']},
            {'role':'user','content':report['input']['prompt_text']}],tokenize=True,add_generation_prompt=True)
        if actual != PROMPT:
            raise RuntimeError('Recorded chat tokenization drift')
        model = AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,
            torch_dtype=torch.float16,attn_implementation='sdpa').to('cuda').eval()
        torch.backends.cuda.matmul.allow_tf32 = False
        inputs = torch.tensor([PROMPT+CONTINUATION[:-1]],device='cuda')
        report['environment']['attention_selected'] = model.config._attn_implementation
        report['logical_kv_bytes'] = 2*model.config.num_hidden_layers*model.config.num_key_value_heads*(model.config.hidden_size//model.config.num_attention_heads)*2*inputs.shape[1]
        gate('after_load')

        @torch.inference_mode()
        def path(cached):
            past, logits, chosen = None, [], []
            for start,end in plan(cached):
                out = model(input_ids=inputs[:,start:end],past_key_values=past,
                    use_cache=cached,logits_to_keep=1)
                logits.append(out.logits[:, -1, :])
                chosen.append(out.logits[:, -1, :].argmax(-1))
                past = out.past_key_values if cached else None
            length = int(past.get_seq_length()) if cached else 0
            return logits,chosen,length

        for cached in [False,True]:
            warm = path(cached)
            torch.cuda.synchronize()
            if not cached:
                reference = torch.cat(warm[0]).float().cpu()
            del warm
        gate('after_warmup')
        baseline_allocation = torch.cuda.memory_allocated()
        for pair in range(PAIRS):
            for cached in ([False,True] if pair%2==0 else [True,False]):
                gate('before_trial')
                torch.cuda.reset_peak_memory_stats()
                start_event,end_event = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                started = time.perf_counter()
                start_event.record()
                values,chosen,length = path(cached)
                end_event.record()
                torch.cuda.synchronize()
                wall_ms = (time.perf_counter()-started)*1000
                interval_ms = start_event.elapsed_time(end_event)
                allocated,reserved = torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved()
                actual = torch.cat(values).float().cpu()
                ids = torch.cat(chosen).tolist()
                row = {'pair':pair,'mode':'cache' if cached else 'no_cache',
                    'wall_ms':wall_ms,'cuda_interval_ms':interval_ms,
                    'peak_allocated_bytes':allocated,'peak_reserved_bytes':reserved,
                    'retained_kv_positions':length,'argmax_ids':ids,
                    'max_abs_logit_difference':float((reference-actual).abs().max()),
                    'logits_allclose':bool(torch.allclose(reference,actual,atol=ATOL,rtol=RTOL)),
                    'argmax_equal':ids == reference.argmax(-1).tolist()}
                report['trials'].append(row)
                del values,chosen,actual,start_event,end_event
                gc.collect()
                gate('after_trial')
                if torch.cuda.memory_allocated() > baseline_allocation+64*1024**2:
                    raise RuntimeError('STOP: allocation failed to recover')
                save()
                print(json.dumps(row),flush=True)
                if not row['logits_allclose'] or not row['argmax_equal']:
                    raise RuntimeError('STOP: output agreement failed')
        report['summary'] = summarize(report['trials'])
        report['profiles'] = []
        # Separate attribution pass, never mixed into timing trials. Full traces
        # remain private; publish only sanitized operator names and scalar values.
        for cached in [False,True]:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                prof_values = path(cached)
                torch.cuda.synchronize()
            del prof_values
            mode = 'cache' if cached else 'no_cache'
            prof.export_chrome_trace(str(output/f'{mode}.private-trace.json'))
            top = sorted(prof.key_averages(),key=lambda event:event.self_device_time_total,reverse=True)[:8]
            report['profiles'].append({'mode':mode,'top_operators':[
                {'name':event.key,'calls':event.count,'self_device_us':event.self_device_time_total} for event in top]})
            del prof
            gate('after_profile')
        report['status'] = 'complete'
    except Exception as error:
        report['status'] = 'failed'
        report['error_type'] = type(error).__name__
        raise
    finally:
        model = tokenizer = inputs = reference = None
        gc.collect()
        torch.cuda.empty_cache()
        report['after_release_allocated_bytes'] = torch.cuda.memory_allocated()
        report['after_release_reserved_bytes'] = torch.cuda.memory_reserved()
        save()
        signal.alarm(0)
    print('RESULT',json.dumps(report),flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--allow-download',action='store_true')
    args = parser.parse_args()
    run(args.output,args.allow_download)
