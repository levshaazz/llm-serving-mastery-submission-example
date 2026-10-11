"""The public notebook's recorded examples must remain CPU-only and exact.

No GPU, installation, weights or network are required. These tests do not replace
the required student GPU protocol or validate a student's measured performance.
"""
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import socket
import sys
import unittest
from unittest.mock import patch
from notebook_static_fixtures import helper_module, recorded_literal

ROOT = Path(__file__).resolve().parents[1]
COURSE_ROOT = ROOT if (ROOT / "seminars").exists() else ROOT.parent
NOTEBOOK = COURSE_ROOT / "seminars/02-gpu-profiling-and-bottlenecks.ipynb"
CAPTURE_SHA256 = "a906a242fb9ab3e781f639ae516e9ff1904b26f111df1830d112b3bf3783b694"


class Topic02RecapNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.code = ["".join(cell["source"]) for cell in cls.notebook["cells"]
                    if cell["cell_type"] == "code"]

    def test_output_free_and_code_compiles(self):
        self.assertEqual(len(self.notebook["cells"]), 57)
        self.assertEqual(len(self.code), 28)
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
                compile("".join(cell["source"]), cell["id"], "exec")

    def test_capture_raw_json_preserves_recorded_bytes(self):
        raw = recorded_literal(self.notebook, "RECORDED_BLOCK")
        self.assertIsInstance(raw, str)
        self.assertEqual(hashlib.sha256(raw.encode("utf-8")).hexdigest(), CAPTURE_SHA256)
        # Hash before parsing: a JavaScript parse/stringify pass can alter the
        # decimal representation of a measured number even if it looks identical.
        captured = json.loads(raw)
        self.assertEqual(captured["method"]["model_forward_calls"], 2)
        self.assertIs(captured["method"]["output_attentions"], False)
        self.assertEqual(captured["prompt"]["last_input_token"]["id"], 198)

    def test_cpu_helper_units_without_network_or_torch(self):
        output = io.StringIO()
        before = set(sys.modules)
        with patch.object(socket, "socket", side_effect=AssertionError("CPU recap must stay offline")), \
             patch.object(socket, "create_connection", side_effect=AssertionError("CPU recap must stay offline")), \
             contextlib.redirect_stdout(output):
            recap = helper_module(self.notebook, 'recap_namespace')
            mechanisms = helper_module(self.notebook, 'mechanism_namespace')
            toy = recap['toy_example']()
            cache_ab = mechanisms['cache_ab'](toy)
            online = mechanisms['online_example']()
        self.assertFalse(any(name == "torch" or name.startswith("torch.")
                             for name in set(sys.modules)-before))
        self.assertNotIn("MODELS", recap)
        self.assertEqual(toy["output"], {"position":2,"index":3,"label":"D"})
        captured = json.loads(recorded_literal(self.notebook, 'RECORDED_BLOCK'))
        self.assertEqual(captured['steps'][1]["selected_token"]["id"], 11)
        self.assertEqual(captured['steps'][1]["selection_position"], 42)
        for index, row in enumerate(toy["stages"]["probabilities"]):
            self.assertAlmostEqual(sum(row), 1)
            self.assertTrue(all(value == 0 for value in row[index+1:]))
        self.assertTrue(any('CPU FP32 reconstruction; not observed SDPA weights' in code for code in self.code))
        self.assertIn('illustrative', recap['__doc__'])
        self.assertIn('invented weights, not Qwen', recap['__doc__'])
        self.assertTrue(any('pip' in code for code in self.code))
        self.assertEqual(cache_ab["baseline"]["qkv_projection_row_calls"],1395)
        self.assertEqual(cache_ab["cached"]["qkv_projection_row_calls"],153)
        self.assertLess(cache_ab["max_abs_logit_difference"],1e-12)
        self.assertAlmostEqual(online["states"][-1]["output"][0],27/7)

    def test_optional_live_is_off_and_preserves_protocol_namespace(self):
        source = next(code for code in self.code if "RUN_CONCRETE_PROMPT_LIVE =" in code)
        self.assertIs(ast.parse(source).body[0].value.value, False)
        tree = ast.parse(source)
        assignments = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)]
        assigned_names = {t.id for n in assignments for t in n.targets if isinstance(t, ast.Name)}
        self.assertNotIn('PROTOCOL', assigned_names)
        self.assertNotIn('MODELS', assigned_names)
        self.assertTrue(any(isinstance(n, ast.If) and isinstance(n.test, ast.Name)
                            and n.test.id == 'RUN_CONCRETE_PROMPT_LIVE' for n in tree.body))


if __name__ == "__main__":
    unittest.main()
