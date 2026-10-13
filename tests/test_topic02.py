import copy
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location("validate_topic02", Path(__file__).resolve().parents[1] / "scripts/validate_topic02.py")
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


def fixture():
    data = {"protocol": "topic02-phase-v2", "status": "complete", "scope": "core", "trials": 5,
            "models": v.MODELS, "cases": list(map(list, v.CASES)),
            "environment": {"dtype": "float16", "transformers": "5.17.0", "attention_requested": "sdpa",
                            "logits_to_keep": 1, "python": "fixture", "torch": "fixture", "cuda": "fixture", "gpu": "fixture"},
            "results": []}
    for model in v.MODELS:
        for batch, prompt, output in v.CASES:
            for repeat in range(5):
                data["results"].append(dict(model=model, batch=batch, input_tokens=prompt, output_tokens=output,
                    repeat=repeat, decode_steps_per_sequence=output-1, generated_tokens=batch*output,
                    retained_tokens_per_sequence=prompt+output-1, prefill_wall_ms=10, prefill_event_ms=9,
                    decode_wall_ms=1000, decode_event_ms=999, decode_aggregate_tokens_per_s=batch*(output-1),
                    decode_mean_step_ms=1000/(output-1), peak_allocated_bytes=200, peak_reserved_bytes=300,
                    prefill_peak_allocated_bytes=100))
    return data


class Topic02Contract(unittest.TestCase):
    def test_valid_synthetic_fixture(self):
        self.assertEqual(v.validate(fixture()), 50)

    def test_partial_and_smoke_rejected(self):
        for key, value in (("status", "failed"), ("scope", "smoke"), ("protocol", "v1")):
            data = fixture(); data[key] = value
            with self.assertRaises(ValueError): v.validate(data)

    def test_missing_duplicate_and_wrong_shape_rejected(self):
        for change in (lambda rows: rows.pop(), lambda rows: rows.append(copy.deepcopy(rows[0])),
                       lambda rows: rows[0].update(output_tokens=512)):
            data = fixture(); change(data["results"])
            with self.assertRaises(ValueError): v.validate(data)

    def test_wrong_token_clock_memory_and_nonfinite_rejected(self):
        for key, value in (("retained_tokens_per_sequence", 160), ("decode_steps_per_sequence", 32),
                           ("decode_aggregate_tokens_per_s", 32), ("decode_mean_step_ms", 9),
                           ("peak_reserved_bytes", 1), ("prefill_event_ms", float("nan")),
                           ("decode_wall_ms", -1)):
            data = fixture(); data["results"][0][key] = value
            with self.assertRaises(ValueError): v.validate(data)

    def test_runtime_mismatch_rejected(self):
        data = fixture(); data["environment"]["logits_to_keep"] = 0
        with self.assertRaises(ValueError): v.validate(data)

    def test_current_version_exact_and_historical_separate(self) -> None:
        """Reject unknown versions; historical validation is an explicit route.

        Returns
        -------
        None
            Synthetic CPU records only, not GPU execution evidence.
        """
        for version in ('4.51.3','4.51.4','5.17','5.17.1','5.16.0','5.17.0+unknown',None,True):
            data=fixture();data['environment']['transformers']=version
            with self.assertRaises(ValueError):v.validate(data)
        legacy_spec=importlib.util.spec_from_file_location('legacy_topic02',
            ROOT/'scripts/validate_topic02_historical.py')
        legacy=importlib.util.module_from_spec(legacy_spec);legacy_spec.loader.exec_module(legacy)
        old=fixture();old['environment']['transformers']='4.51.3'
        self.assertEqual(legacy.validate(old),50)
        with self.assertRaises(ValueError):legacy.validate(fixture())


if __name__ == "__main__": unittest.main()
