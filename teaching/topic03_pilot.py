"""Bounded Topic 03 pilot infrastructure, separate from student TODO functions.

Import is CPU-only and standard-library-only. GPU libraries load inside run().
An output artifact is teaching evidence, never a judge or production benchmark.
Pedagogical influences: MIT EfficientML Lab 4 controlled comparisons;
CMU Deep Learning Systems short student functions in one evolving project.
Original implementation, not copied lab code.
"""
import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import signal
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

PROTOCOL = 'topic03-engineering-pilot-v1'
MODEL_ID = 'Qwen/Qwen2.5-0.5B-Instruct'
REVISION = '7ae557604adf67be50417f59c2c2f167def9a775'
LABELS = ('billing', 'access', 'bug')
SYSTEM = ('Classify the support ticket. Reply with exactly one lowercase label: billing, access, or bug. '
          'billing = payments, invoices, refunds; access = login or account permissions; '
          'bug = software malfunction after successful login. No explanation.')
# Handwritten synthetic teaching cases, NOT a sample of a real support workload.
CASES = [
    {'id': 'b1', 'text': 'I was charged twice for my monthly subscription. Please refund the duplicate.', 'label': 'billing'},
    {'id': 'b2', 'text': 'Please send an invoice for the payment we made last week.', 'label': 'billing'},
    {'id': 'b3', 'text': 'My credit card expired and the subscription payment was declined.', 'label': 'billing'},
    {'id': 'b4', 'text': 'I cancelled the subscription but a new charge appeared on my bank statement.', 'label': 'billing'},
    {'id': 'a1', 'text': 'I forgot my password and cannot sign in to my account.', 'label': 'access'},
    {'id': 'a2', 'text': 'Our new employee needs permission to open the team workspace.', 'label': 'access'},
    {'id': 'a3', 'text': 'I replaced my phone and no longer have my two-factor login codes.', 'label': 'access'},
    {'id': 'a4', 'text': 'My account is locked after too many failed sign-in attempts.', 'label': 'access'},
    {'id': 'g1', 'text': 'I can log in, but clicking Export crashes the application.', 'label': 'bug'},
    {'id': 'g2', 'text': 'After successful login, the Save button does nothing and my edits disappear.', 'label': 'bug'},
    {'id': 'g3', 'text': 'I am signed in. The search page shows an error for every search query.', 'label': 'bug'},
    {'id': 'g4', 'text': 'My account works, but every image I upload is displayed upside down.', 'label': 'bug'},
]
# Public transfer exercise; do not use these cases to tune the core prompt.
# They are held out of the default RUN, not a secret or statistically independent test.
TRANSFER_CASES = [
    {'id': 't1', 'text': 'Please correct the company name on the invoice for our paid plan.', 'label': 'billing'},
    {'id': 't2', 'text': 'The yearly plan renewed yesterday. Can the payment be refunded?', 'label': 'billing'},
    {'id': 't3', 'text': 'The invitation to join my team expired before I could accept it.', 'label': 'access'},
    {'id': 't4', 'text': 'I know my password, but the login page says my account is disabled.', 'label': 'access'},
    {'id': 't5', 'text': 'I am logged in, but the calendar always displays the wrong day for saved events.', 'label': 'bug'},
    {'id': 't6', 'text': 'Login succeeds. Downloading a report produces an empty file instead of the report.', 'label': 'bug'},
]
TIMING_PROMPT = 'Continue the sequence with comma-separated integers and no explanation: 2, 4, 6,'
PINS = {'transformers': '4.51.3', 'accelerate': '1.6.0', 'bitsandbytes': '0.50.2',
        'huggingface_hub': '0.30.2', 'safetensors': '0.5.3'}


def utc():
    return datetime.now(timezone.utc).isoformat()


def checkpoint(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def score_output(output, expected):
    predicted = output.strip()  # Only surrounding whitespace may be ignored.
    valid = predicted in LABELS
    return {'predicted_label': predicted if valid else None,
            'valid_label': valid, 'correct': valid and predicted == expected}


def summary(values):
    if not values or any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError('nonempty positive finite timings required')
    return {'median_ms': 1000*statistics.median(values),
            'minimum_ms': 1000*min(values), 'maximum_ms': 1000*max(values)}


def resource_sample(require_idle=False, startup=False):
    """Headroom checks, NOT an OS memory sandbox. No process/account paths logged."""
    lines = Path('/proc/meminfo').read_text().splitlines()
    host_available = int(next(row.split()[1] for row in lines if row.startswith('MemAvailable:')))*1024
    host_total = int(next(row.split()[1] for row in lines if row.startswith('MemTotal:')))*1024
    query = subprocess.check_output(['nvidia-smi', '--id=0',
        '--query-gpu=memory.total,memory.used,memory.free', '--format=csv,noheader,nounits'],
        text=True, timeout=10).strip()
    total, used, free = [int(part.strip())*2**20 for part in query.split(',')]
    if host_available < (5 if startup else 2)*2**30 or free < (5 if startup else 1)*2**30:
        raise RuntimeError('STOP: pilot headroom insufficient; diagnose, do not lower gates')
    if require_idle:
        pids = subprocess.check_output(['nvidia-smi', '--id=0', '--query-compute-apps=pid',
                                       '--format=csv,noheader,nounits'], text=True, timeout=10).split()
        if any(pid != str(os.getpid()) for pid in pids):
            raise RuntimeError('STOP: another GPU compute process is active')
    return {'host_total_bytes': host_total, 'host_available_bytes': host_available,
            'device_total_bytes': total, 'device_used_bytes': used, 'device_free_bytes': free}


def snapshot_path():
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    kwargs = {'repo_id': MODEL_ID, 'revision': REVISION,
              'allow_patterns': ['*.json', '*.safetensors', '*.txt', '*.model']}
    if os.environ.get('LSM_MODEL_CACHE'):
        kwargs['cache_dir'] = os.environ['LSM_MODEL_CACHE']
    try:
        return snapshot_download(**kwargs, local_files_only=True), 'reused_local_snapshot'
    except LocalEntryNotFoundError:
        if os.environ.get('HF_HUB_OFFLINE') == '1':
            raise RuntimeError('Pinned snapshot missing in offline mode; preserve existing caches') from None
        # Conservative additional-download headroom for the 0.5B checkpoint only.
        disk_root = Path(kwargs.get('cache_dir') or os.environ.get('HF_HOME', Path.home()/'.cache/huggingface'))
        while not disk_root.exists():
            disk_root = disk_root.parent
        if shutil.disk_usage(disk_root).free < 3*2**30:
            raise RuntimeError('STOP: less than 3 GiB free cache-disk headroom')
        return snapshot_download(**kwargs), 'downloaded_missing_pinned_snapshot'


def profile_once(torch, model, ids, generate):
    """Operator attribution, not a hardware bandwidth proof; overlaps are possible."""
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as profile:
            generate(model, ids, 24, True)
        events = profile.key_averages()
        ranked = sorted(events, key=lambda event: getattr(event, 'device_time_total', 0), reverse=True)
        rows = [{'operator': event.key, 'calls': event.count,
                 'cpu_total_us': event.cpu_time_total,
                 'device_total_us': getattr(event, 'device_time_total', 0)} for event in ranked[:12]]
        if not any(row['device_total_us'] > 0 for row in rows):
            return {'status': 'unavailable', 'error_type': 'NoDeviceAttribution',
                    'limitation': 'Profiler did not retain CUDA attribution; no kernel-cause claim is supported.'}
        return {'status': 'complete', 'output_tokens': 24, 'rows': rows,
                'limitation': 'Inclusive operator times may overlap; not disjoint kernels or measured HBM traffic.'}
    except torch.cuda.OutOfMemoryError:
        raise  # OOM is a failed run, never a harmless missing-profiler feature.
    except RuntimeError as exc:
        if not any(term in str(exc).lower() for term in ['cupti', 'profiler', 'kineto', 'permission', 'profiling', 'not supported']):
            raise
        # Only recognized profiler availability failures are an explicit partial result.
        return {'status': 'unavailable', 'error_type': type(exc).__name__,
                'limitation': 'No profile-based causal interpretation is supported by this attempt.'}


def run(output, lengths=(24, 96), trials=5, order=('fp16', 'nf4'), timeout_s=600):
    from importlib.metadata import version
    output = Path(output)
    if output.exists():
        raise FileExistsError('Preserve prior attempt; choose a fresh output path')
    if tuple(lengths) != (24, 96) or trials != 5 or set(order) != {'fp16', 'nf4'} or len(order) != 2:
        raise ValueError('Pilot contract: lengths 24/96, five trials, each format exactly once')
    if not 120 <= timeout_s <= 600:
        raise ValueError('Choose a 120–600 second wall budget')
    if any(version(package) != wanted for package, wanted in PINS.items()):
        raise RuntimeError('Use the exact pilot package pins; Torch/CUDA are recorded from the host')
    first_sample = resource_sample(require_idle=True, startup=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = {'contract': PROTOCOL, 'status': 'running', 'utc_started': utc(),
            'model_id': MODEL_ID, 'revision': REVISION, 'format_order': list(order),
            'lengths': list(lengths), 'trials': trials, 'formats': {},
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'runtime_pins': PINS, 'dataset_kind': 'handwritten synthetic teaching cases',
            'cases': CASES, 'quality_cap': 12, 'resource_samples': [first_sample],
            'quantization': {'quant_type': 'nf4', 'compute_dtype': 'float16', 'double_quant': False},
            'clock': 'CUDA-synchronized generate wall; excludes tokenization, input transfer and model load; includes host dispatch; not HTTP latency',
            'limitations': ['one model, GPU and batch=1; two sequential format loads',
                            'format order can confound a small observed timing difference',
                            'five raw trials per condition, no tail percentile or population speed claim',
                            '12 synthetic natural samples do not certify production quality',
                            'allocator counters are not total process or device VRAM',
                            'fixed-length continuation is a synthetic timing workload, never scored as quality',
                            'sampled headroom is not an OS memory limit or continuous peak monitor',
                            'reserved pools persist across warmups and lengths; per-row counters are not independent fresh allocators',
                            'the format intervention changes representation AND execution path, not bit width alone']}
    checkpoint(output, data)
    model = None
    torch = None
    previous_alarm = signal.getsignal(signal.SIGALRM)
    previous_term = signal.getsignal(signal.SIGTERM)
    def stop(signum, frame):
        raise TimeoutError('Pilot wall budget or external termination reached')
    signal.signal(signal.SIGALRM, stop)
    signal.signal(signal.SIGTERM, stop)
    signal.alarm(timeout_s)
    begin_all = time.perf_counter()
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required; a recorded example is not your execution')
        data['runtime'] = {'torch': torch.__version__, 'cuda': torch.version.cuda,
                           'gpu': torch.cuda.get_device_name(0),
                           'driver': subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=driver_version',
                              '--format=csv,noheader'], text=True, timeout=10).strip()}
        snapshot, data['cache_action'] = snapshot_path()
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        def tokenized(messages):
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
            if not isinstance(ids, torch.Tensor):
                ids = ids['input_ids']
            return rendered, ids.to('cuda')
        def generate(model, ids, cap, forced):
            kwargs = {'do_sample': False, 'max_new_tokens': cap, 'use_cache': True,
                      'pad_token_id': tokenizer.eos_token_id}
            if forced:
                kwargs['min_new_tokens'] = cap
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                result = model.generate(ids, **kwargs)
            torch.cuda.synchronize()
            wall_s = time.perf_counter()-started
            selected = result[0, ids.shape[-1]:].tolist()
            del result
            if not selected or (forced and len(selected) != cap):
                raise RuntimeError('Generation count violates the declared workload')
            return {'wall_s': wall_s, 'output_tokens': len(selected), 'output_token_ids': selected,
                    'output': tokenizer.decode(selected, skip_special_tokens=True)}
        for label in order:
            data['resource_samples'].append(resource_sample())
            gc.collect()
            torch.cuda.empty_cache()
            before = torch.cuda.memory_allocated()
            quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=False) if label == 'nf4' else None
            loaded = time.perf_counter()
            model = AutoModelForCausalLM.from_pretrained(snapshot, local_files_only=True,
                    torch_dtype=torch.float16, quantization_config=quant, device_map={'': 0},
                    attn_implementation='sdpa').eval()
            torch.cuda.synchronize()
            entry = {'status': 'running', 'load_s': time.perf_counter()-loaded,
                     'baseline_allocated_bytes': before,
                     'quantized_linear_modules': sum(type(module).__name__ == 'Linear4bit' for module in model.modules()),
                     'timing_trials': [], 'quality_samples': [], 'summaries': {}}
            # Actual storage, before clocks; no per-layer weight values exported.
            ledger = []
            state = None
            for name, module in model.named_modules():
                if type(module).__name__ != 'Linear4bit':
                    continue
                state = module.weight.quant_state
                ledger.append(dict(module=name, shape_out_in=list(state.shape),
                    blocksize=state.blocksize, quant_type=state.quant_type,
                    absmax_dtype=str(state.absmax.dtype), absmax_count=state.absmax.numel(),
                    absmax_bytes=state.absmax.numel()*state.absmax.element_size(),
                    packed_dtype=str(module.weight.dtype),
                    packed_bytes=module.weight.numel()*module.weight.element_size(),
                    nested=bool(state.nested), compute_dtype=str(module.compute_dtype)))
            entry['quantization_ledger'] = ledger
            data['storage_ledger_revision'] = 'module-storage-v1'
            del module, state  # Do not keep the last module alive past model cleanup.
            eos = model.generation_config.eos_token_id
            entry['eos_token_ids'] = eos if isinstance(eos, list) else [eos]
            data['formats'][label] = entry
            rendered, timing_ids = tokenized([{'role': 'user', 'content': TIMING_PROMPT}])
            entry['timing_input'] = {'prompt': TIMING_PROMPT, 'rendered_chat': rendered,
                                      'input_token_ids': timing_ids[0].tolist(), 'prompt_tokens': timing_ids.shape[-1]}
            # One excluded warmup for EACH length/format condition.
            for length in lengths:
                generate(model, timing_ids, length, True)
            for trial in range(trials):
                for length in (lengths if trial % 2 == 0 else tuple(reversed(lengths))):
                    data['resource_samples'].append(resource_sample())
                    torch.cuda.reset_peak_memory_stats()
                    row = generate(model, timing_ids, length, True)
                    row.update(trial=trial, length=length, sequence=len(entry['timing_trials']),
                               peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                               peak_reserved_bytes=torch.cuda.max_memory_reserved())
                    entry['timing_trials'].append(row)
                    checkpoint(output, data)
            for length in lengths:
                rows = [row for row in entry['timing_trials'] if row['length'] == length]
                entry['summaries'][str(length)] = {**summary([row['wall_s'] for row in rows]),
                    'peak_allocated_bytes': max(row['peak_allocated_bytes'] for row in rows),
                    'peak_reserved_bytes': max(row['peak_reserved_bytes'] for row in rows)}
            for case in CASES:
                resource_sample()
                rendered_quality, quality_ids = tokenized([{'role': 'system', 'content': SYSTEM},
                                                            {'role': 'user', 'content': case['text']}])
                row = generate(model, quality_ids, 12, False)
                row.update(case_id=case['id'], expected_label=case['label'],
                           rendered_chat=rendered_quality, input_token_ids=quality_ids[0].tolist(),
                           stop_reason='eos' if row['output_token_ids'][-1] in entry['eos_token_ids'] else 'length',
                           **score_output(row['output'], case['label']))
                entry['quality_samples'].append(row)
                del quality_ids
                checkpoint(output, data)
            entry['quality_summary'] = {'count': len(CASES),
                'correct_count': sum(row['correct'] for row in entry['quality_samples']),
                'invalid_count': sum(not row['valid_label'] for row in entry['quality_samples'])}
            # Profiles come AFTER all timing rows; do not mix instrumented timings in the table.
            resource_sample()
            entry['profile'] = profile_once(torch, model, timing_ids, generate)
            del timing_ids, model
            model = None
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            entry['post_cleanup_allocated_bytes'] = torch.cuda.memory_allocated()
            if entry['post_cleanup_allocated_bytes'] > before + 64*2**20:
                raise RuntimeError('STOP: model allocation did not recover after cleanup')
            entry.update(status='complete', cleanup_verified=True)
            checkpoint(output, data)
        data['status'] = 'complete'
        data['resource_samples'].append(resource_sample())
    except BaseException as exc:
        data.update(status='failed', error_type=type(exc).__name__)
        raise
    finally:
        signal.alarm(0)
        if model is not None:
            del model
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            data['final_allocated_bytes'] = torch.cuda.memory_allocated()
        data.update(utc_finished=utc(), elapsed_s=time.perf_counter()-begin_all)
        checkpoint(output, data)
        signal.signal(signal.SIGALRM, previous_alarm)
        signal.signal(signal.SIGTERM, previous_term)
    validate_artifact(data)
    return data


def validate_artifact(data):
    """CPU structural check. Passing is not semantic quality or causality certification."""
    def require(condition, reason):
        if not condition:
            raise ValueError(reason)
    def number(value):
        return type(value) in (int, float) and math.isfinite(value) and value > 0
    require(data.get('contract') == PROTOCOL and data.get('status') == 'complete', 'complete pilot protocol')
    require(data.get('model_id') == MODEL_ID and data.get('revision') == REVISION, 'pinned model identity')
    require(data.get('lengths') == [24, 96] and data.get('trials') == 5, 'one-factor timing matrix')
    require(data.get('runtime_pins') == PINS and data.get('cases') == CASES, 'pinned dependencies and dataset')
    require(data.get('quality_cap') == 12 and data.get('quantization') == {
        'quant_type': 'nf4', 'compute_dtype': 'float16', 'double_quant': False}, 'quantization and quality caps')
    require(set(data.get('formats', {})) == {'fp16', 'nf4'}, 'paired treatments')
    require(data.get('format_order') in [['fp16', 'nf4'], ['nf4', 'fp16']], 'recorded treatment order')
    require(isinstance(data.get('source_sha256'), str) and len(data['source_sha256']) == 64, 'source hash')
    require(all(isinstance(data.get('runtime', {}).get(key), str) and data['runtime'][key]
                for key in ['torch', 'cuda', 'gpu', 'driver']), 'runtime identity')
    require(number(data.get('elapsed_s')) and type(data.get('final_allocated_bytes')) is int, 'execution/cleanup')
    paired_inputs = []
    ledger_required = data.get('storage_ledger_revision') == 'module-storage-v1' or data.get('source_sha256') == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for label, entry in data['formats'].items():
        require(entry.get('status') == 'complete' and entry.get('cleanup_verified') is True, 'format completed')
        require(number(entry.get('load_s')), 'load clock')
        require(entry.get('quantized_linear_modules') == 0 if label == 'fp16' else
                type(entry.get('quantized_linear_modules')) is int and entry['quantized_linear_modules'] > 0,
                'actual quantized module treatment')
        if ledger_required or 'quantization_ledger' in entry:
            ledger = entry.get('quantization_ledger')
            require(isinstance(ledger, list) and len(ledger) == entry['quantized_linear_modules'], 'complete module storage ledger')
            require(len({r.get('module') for r in ledger}) == len(ledger), 'unique storage ledger modules')
            for row in ledger:
                shape = row.get('shape_out_in', [])
                require(len(shape) == 2 and all(type(n) is int and n > 0 for n in shape), 'storage weight shape')
                count = math.prod(shape)
                require(row.get('blocksize') == 64 and row.get('quant_type') == 'nf4' and row.get('nested') is False,
                        'actual NF4 block size and double-quant setting')
                require(row.get('absmax_dtype') == 'torch.float32' and row.get('absmax_count') == math.ceil(count/64) and
                        row.get('absmax_bytes') == 4*math.ceil(count/64), 'actual FP32 absmax storage')
                require(row.get('packed_dtype') == 'torch.uint8' and row.get('packed_bytes') == math.ceil(count/2) and
                        row.get('compute_dtype') == 'torch.float16', 'actual packed weight storage and compute dtype')
        require(type(entry.get('baseline_allocated_bytes')) is int and entry['baseline_allocated_bytes'] >= 0 and
                type(entry.get('post_cleanup_allocated_bytes')) is int and 0 <= entry['post_cleanup_allocated_bytes'] <=
                entry['baseline_allocated_bytes'] + 64*2**20, 'model cleanup')
        timing_input = entry.get('timing_input', {})
        require(timing_input.get('prompt') == TIMING_PROMPT and
                timing_input.get('prompt_tokens') == len(timing_input.get('input_token_ids', [])) > 0,
                'literal timing prompt and tokens')
        rows = entry.get('timing_trials', [])
        require(len(rows) == 10, 'ten raw timing trials per format')
        require([(row.get('trial'), row.get('length')) for row in rows] ==
                [(trial, length) for trial in range(5) for length in ([24, 96] if trial % 2 == 0 else [96, 24])],
                'alternating length order, no duplicated trial')
        for index, row in enumerate(rows):
            require(row.get('sequence') == index and number(row.get('wall_s')), 'raw sequence and clock')
            require(row.get('output_tokens') == row['length'] == len(row.get('output_token_ids', [])), 'fixed output length')
            require(type(row.get('peak_allocated_bytes')) is int and type(row.get('peak_reserved_bytes')) is int and
                    0 < row['peak_allocated_bytes'] <= row['peak_reserved_bytes'], 'allocator, not device counters')
        for length in [24, 96]:
            selected = [row for row in rows if row['length'] == length]
            expected = {**summary([row['wall_s'] for row in selected]),
                'peak_allocated_bytes': max(row['peak_allocated_bytes'] for row in selected),
                'peak_reserved_bytes': max(row['peak_reserved_bytes'] for row in selected)}
            require(entry.get('summaries', {}).get(str(length)) == expected, 'raw-to-summary arithmetic')
        quality = entry.get('quality_samples', [])
        require(len(quality) == len(CASES), 'twelve separate natural outputs')
        require(isinstance(entry.get('eos_token_ids'), list) and len(entry['eos_token_ids']) > 0 and
                all(type(token) is int and token >= 0 for token in entry['eos_token_ids']), 'EOS identity')
        for case, row in zip(CASES, quality):
            require(row.get('case_id') == case['id'] and row.get('expected_label') == case['label'], 'quality identity')
            require(number(row.get('wall_s')) and 0 < row.get('output_tokens', 0) <= 12 and
                    row['output_tokens'] == len(row.get('output_token_ids', [])), 'natural output budget')
            require(row.get('stop_reason') == ('eos' if row['output_token_ids'][-1] in entry['eos_token_ids'] else 'length') and
                    (row['stop_reason'] == 'eos' or row['output_tokens'] == 12), 'natural stop reason')
            require(all(row.get(key) == value for key, value in score_output(row.get('output', ''), case['label']).items()),
                    'exact label scoring')
        require(entry.get('quality_summary') == {'count': len(CASES),
            'correct_count': sum(row['correct'] for row in quality),
            'invalid_count': sum(not row['valid_label'] for row in quality)}, 'quality aggregate')
        profile = entry.get('profile', {})
        require(profile.get('status') in ('complete', 'unavailable'), 'explicit profiler outcome')
        if profile['status'] == 'complete':
            require(profile.get('output_tokens') == 24 and isinstance(profile.get('rows'), list) and
                    0 < len(profile['rows']) <= 12, 'separate bounded profile')
        paired_inputs.append((timing_input, [(row['rendered_chat'], row['input_token_ids']) for row in quality]))
    require(paired_inputs[0] == paired_inputs[1], 'identical tokenized inputs across treatments')
    require(isinstance(data.get('resource_samples'), list) and len(data['resource_samples']) == 24,
            'preflight, format starts, twenty trials, final headroom')
    for sample in data['resource_samples']:
        require(all(type(sample.get(key)) is int and sample[key] >= 0 for key in
                    ['host_total_bytes', 'host_available_bytes', 'device_total_bytes', 'device_used_bytes', 'device_free_bytes']),
                'resource sample bytes')
        require(sample['host_available_bytes'] >= 2*2**30 and sample['device_free_bytes'] >= 2**30,
                'runtime headroom')
    require(data['resource_samples'][0]['host_available_bytes'] >= 5*2**30 and
            data['resource_samples'][0]['device_free_bytes'] >= 5*2**30, 'startup headroom')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--lengths', nargs=2, type=int, default=[24, 96])
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--format-order', nargs=2, choices=['fp16', 'nf4'], default=['fp16', 'nf4'])
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--validate', type=Path, help='CPU-only validation; no CUDA/package import')
    args = parser.parse_args()
    if args.validate:
        validate_artifact(json.loads(args.validate.read_text(encoding='utf-8')))
        print('Pilot artifact is structurally complete; not a quality or performance certification.')
    elif args.output:
        result = run(args.output, args.lengths, args.trials, args.format_order, args.timeout)
        print(json.dumps({'status': result['status'], 'elapsed_s': result['elapsed_s'],
                          'summaries': {key: entry['summaries'] for key, entry in result['formats'].items()}}, indent=2))
    else:
        parser.error('--output (GPU) or --validate (CPU) is required')


if __name__ == '__main__':
    main()
