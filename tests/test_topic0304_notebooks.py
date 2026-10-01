"""Portable CPU/offline first cells and artifact validators; never invokes a GPU helper."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import socket
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
# In the private source, notebooks live one level above submission-template.
NOTEBOOK_ROOT=ROOT if (ROOT/'seminars').exists() else ROOT.parent


class Topic0304NotebookTests(unittest.TestCase):
    def load(self,topic):
        path=NOTEBOOK_ROOT/'seminars'/({'03':'03-quantization-tradeoffs.ipynb','04':'04-vllm-openai-serving.ipynb'}[topic])
        data=json.loads(path.read_text(encoding='utf-8'))
        for cell in data['cells']:
            if cell['cell_type']=='code':
                self.assertEqual(cell['outputs'],[]); self.assertIsNone(cell['execution_count'])
                compile(''.join(cell['source']),cell['id'],'exec')
        return [''.join(c['source']) for c in data['cells'] if c['cell_type']=='code']

    def validator(self,topic):
        path=ROOT/'scripts'/f'validate_topic{topic}.py'
        spec=importlib.util.spec_from_file_location('validator_'+topic,path)
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        return module.validate_artifact

    def execute_cpu(self,topic):
        code=self.load(topic); namespace={}; before=set(sys.modules)
        with patch.object(socket,'socket',side_effect=AssertionError('CPU replay cannot connect')), \
             patch.object(socket,'create_connection',side_effect=AssertionError('CPU replay cannot connect')), \
             contextlib.redirect_stdout(io.StringIO()):
            for source in code[:3]: exec(source,namespace)
        self.assertFalse(any(n=='torch' or n.startswith('torch.') for n in set(sys.modules)-before))
        self.assertNotIn('lab_namespace',namespace)
        self.assertIn('pip',code[3])
        return namespace

    def test_topic03_offline_and_real_artifact(self):
        ns=self.execute_cpu('03')
        self.assertEqual(ns['TOY']['packed_hex'],['e9','71','c1','84'])
        self.assertTrue(self.validator('03')(ns['RECORDED_QUANTIZATION']))

    def test_topic04_offline_and_real_artifact(self):
        ns=self.execute_cpu('04')
        self.assertEqual(ns['FIXTURE']['complete']['content'],'Café ☕.')
        self.assertEqual(ns['REPLAY']['status'],'complete')
        self.assertTrue(self.validator('04')(ns['RECORDED_SERVICE']))

    def test_quantization_negative_artifacts(self):
        original=self.execute_cpu('03')['RECORDED_QUANTIZATION']; validate=self.validator('03')
        mutations=[lambda x:x.update(status='running'),lambda x:x.update(revision='main'),
            lambda x:x.update(transformers='unknown'),lambda x:x.update(quantization={}),
            lambda x:x['formats']['nf4_w4a16'].update(quantized_linear_modules=0),
            lambda x:x['formats']['fp16'].update(cleanup_verified=False),
            lambda x:x['formats']['fp16'].update(peak_allocated_bytes=float('nan')),
            lambda x:x['formats']['fp16']['rows'].pop(),
            lambda x:x['formats']['fp16']['rows'][0]['timing_trials'].pop(),
            lambda x:x['formats']['fp16']['rows'][0]['timing_trials'][0].update(wall_s=float('inf')),
            lambda x:x['formats']['fp16']['rows'][0]['timing_trials'][0].update(output_tokens=23),
            lambda x:x['formats']['fp16']['rows'][0]['quality_sample'].update(stop_reason='fixed_length'),
            lambda x:x['formats']['nf4_w4a16']['rows'][0]['quality_sample']['input_token_ids'].__setitem__(0,999),
            lambda x:x['formats']['fp16']['rows'][0].update(p95_ttft_s=999),
            lambda x:x['formats']['fp16'].update(load_s=True),
            lambda x:x['formats']['fp16'].update(eos_token_ids=[True]),
            lambda x:x['formats']['fp16']['rows'][0]['timing_trials'][0].update(ttft_s=True,wall_s=True,post_first_s_per_token=0),
            lambda x:x['resource_samples'][0].update(device_used_mib=True),
            lambda x:x['resource_samples'][0].update(available_ram_gib=23)]
        for mutate in mutations:
            bad=copy.deepcopy(original); mutate(bad)
            with self.assertRaises((ValueError,KeyError,TypeError)): validate(bad)

    def test_serving_negative_artifacts(self):
        original=self.execute_cpu('04')['RECORDED_SERVICE']; validate=self.validator('04')
        mutations=[lambda x:x.update(status='failed'),lambda x:x.update(vllm='unknown'),
            lambda x:x.update(served_name='wrong'),lambda x:x.update(advertised_models=[]),
            lambda x:x['raw_sse'].update(content='invented'),lambda x:x['raw_sse'].update(done=False),
            lambda x:x['raw_sse']['chunks'].pop(),lambda x:x['raw_sse'].update(http_status=500),
            lambda x:x['raw_prompt'].update(prompt_tokens=999),
            lambda x:x['request_prompt']['messages'][0].update(content='not the recorded request'),
            lambda x:x['nonstream']['usage'].update(total_tokens=0),
            lambda x:x['sdk_stream'].update(first_content_s=-1),
            lambda x:x['teardown'].update(owned_process_group_stopped=False),
            lambda x:x['negative']['wrong_model'].update(status=200),
            lambda x:x['raw_sse']['request'].update(max_tokens=32),
            lambda x:x['raw_sse']['request'].update(temperature=1),
            lambda x:x['raw_sse']['request']['stream_options'].update(include_usage=False),
            lambda x:x['nonstream']['request'].update(model='wrong'),
            lambda x:x['nonstream']['usage'].update(completion_tokens=float(x['nonstream']['usage']['completion_tokens'])),
            lambda x:x['limits'].update(gpu_memory_utilization=.9),
            lambda x:x['launch_args'].__setitem__(5,'0.0.0.0'),
            lambda x:x.update(startup_s=True),
            lambda x:x['teardown'].update(owned_process_group_stopped=1),
            lambda x:x['resource_samples'][-1].update(device_used_mib=x['resource_samples'][0]['device_used_mib']+65),
            lambda x:x['raw_sse']['request'].update(temperature=False),
            lambda x:x['raw_sse']['request'].update(stream=1),
            lambda x:x['raw_sse']['request']['stream_options'].update(include_usage=1),
            lambda x:x['negative']['wrong_model'].update(status=404.0),
            lambda x:x['negative']['over_context'].update(rendered_prompt_tokens=2329.0)]
        for mutate in mutations:
            bad=copy.deepcopy(original); mutate(bad)
            with self.assertRaises((ValueError,KeyError,TypeError)): validate(bad)


if __name__=='__main__': unittest.main()
