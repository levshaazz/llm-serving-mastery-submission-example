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
        self.assertEqual(len(self.notebook["cells"]), 26)
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
                compile("".join(cell["source"]), cell["id"], "exec")

    def test_capture_raw_json_preserves_recorded_bytes(self):
        assignment = ast.parse(self.code[3]).body[0]
        self.assertEqual(assignment.targets[0].id, "RECORDED_BLOCK")
        raw = assignment.value.args[0].value
        self.assertIsInstance(raw, str)
        self.assertEqual(hashlib.sha256(raw.encode("utf-8")).hexdigest(), CAPTURE_SHA256)
        # Hash before parsing: a JavaScript parse/stringify pass can alter the
        # decimal representation of a measured number even if it looks identical.
        captured = json.loads(raw)
        self.assertEqual(captured["method"]["model_forward_calls"], 2)
        self.assertIs(captured["method"]["output_attentions"], False)
        self.assertEqual(captured["prompt"]["last_input_token"]["id"], 198)

    def test_first_four_cells_execute_without_network_or_torch(self):
        namespace, output = {}, io.StringIO()
        before = set(sys.modules)
        with patch.object(socket, "socket", side_effect=AssertionError("CPU recap must stay offline")), \
             patch.object(socket, "create_connection", side_effect=AssertionError("CPU recap must stay offline")), \
             contextlib.redirect_stdout(output):
            for source in self.code[:4]:
                exec(source, namespace)
            exec(self.code[3].replace("STEP = 0", "STEP = 1", 1), namespace)
        self.assertFalse(any(name == "torch" or name.startswith("torch.")
                             for name in set(sys.modules)-before))
        self.assertNotIn("MODELS", namespace)
        self.assertEqual(namespace["TOY"]["output"], {"position":2,"index":3,"label":"D"})
        self.assertEqual(namespace["block_step"]["selected_token"]["id"], 11)
        self.assertEqual(namespace["block_step"]["selection_position"], 42)
        for index, row in enumerate(namespace["TOY"]["stages"]["probabilities"]):
            self.assertAlmostEqual(sum(row), 1)
            self.assertTrue(all(value == 0 for value in row[index+1:]))
        self.assertIn("CPU FP32 reconstruction; not observed SDPA weights", output.getvalue())
        self.assertIn("ILLUSTRATIVE", output.getvalue())
        self.assertIn("pip", self.code[4])

    def test_optional_live_is_off_and_preserves_protocol_namespace(self):
        source = next(code for code in self.code if "RUN_CONCRETE_PROMPT_LIVE =" in code)
        self.assertIs(ast.parse(source).body[0].value.value, False)
        sentinel = {"unchanged":True}
        namespace = {"PROTOCOL":"topic02-phase-v2","MODELS":sentinel}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(source, namespace)
        self.assertEqual(namespace["PROTOCOL"], "topic02-phase-v2")
        self.assertIs(namespace["MODELS"], sentinel)
        self.assertNotIn("LIVE_PROMPT", namespace)


if __name__ == "__main__":
    unittest.main()
