from __future__ import annotations

import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("validator", ROOT / "scripts" / "validate_submission.py")
validator = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(validator)


class ValidatorTests(unittest.TestCase):
    def copy_example(self) -> Path:
        destination = Path(tempfile.mkdtemp()) / "submission"
        shutil.copytree(ROOT, destination, ignore=shutil.ignore_patterns(".git", "__pycache__"))
        metadata = destination / "submission.yaml"
        metadata.write_text(metadata.read_text().replace("your-github-handle", "student"), encoding="utf-8")
        self.addCleanup(shutil.rmtree, destination.parent)
        return destination

    def test_example_is_valid(self):
        self.assertEqual([], validator.validate(self.copy_example()))

    def test_revision_mismatch_is_rejected(self):
        root = self.copy_example()
        serve = root / "serve.sh"
        serve.write_text(serve.read_text().replace("aa8e72537993ba99e69dfaafa59ed015b17504d1", "0" * 40), encoding="utf-8")
        self.assertTrue(any("MODEL_REVISION" in error for error in validator.validate(root)))

    def test_missing_served_name_is_rejected(self):
        root = self.copy_example()
        serve = root / "serve.sh"
        serve.write_text(serve.read_text().replace("--served-model-name submission", "--served-model-name wrong"), encoding="utf-8")
        self.assertTrue(any("served model name" in error for error in validator.validate(root)))


if __name__ == "__main__":
    unittest.main()
