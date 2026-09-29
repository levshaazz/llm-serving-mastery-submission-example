#!/usr/bin/env python3
"""Fast, GPU-free validation of the public submission contract."""

from __future__ import annotations

import argparse
import re
import stat
import sys
import tomllib
from pathlib import Path

import yaml


REQUIRED = {
    "README.md",
    "JOURNAL.md",
    "submission.yaml",
    "prepare.sh",
    "serve.sh",
    "pyproject.toml",
    "uv.lock",
}
REVISION = re.compile(r"^[0-9a-f]{40}$")
SECRET_PATTERNS = {
    "GitHub classic token": re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    "GitHub fine-grained token": re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    "Hugging Face token": re.compile(r"hf_[A-Za-z0-9]{20,}"),
    "OpenAI-style key": re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
}


def assignment(text: str, name: str) -> str | None:
    match = re.search(rf"(?m)^{re.escape(name)}=([^\s#]+)", text)
    return match.group(1).strip("'\"") if match else None


def validate(root: Path, *, allow_placeholder_author: bool = False) -> list[str]:
    errors: list[str] = []
    missing = sorted(name for name in REQUIRED if not (root / name).is_file())
    if missing:
        return [f"missing required file: {name}" for name in missing]

    metadata = yaml.safe_load((root / "submission.yaml").read_text(encoding="utf-8")) or {}
    for key in ("author", "model", "model_revision", "stack", "notes"):
        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
            errors.append(f"submission.yaml: {key} must be a non-empty string")
    if not allow_placeholder_author and metadata.get("author") in {"your-github-handle", "TODO", "changeme"}:
        errors.append("submission.yaml: replace the placeholder author")
    revision = metadata.get("model_revision", "")
    if not REVISION.fullmatch(revision):
        errors.append("submission.yaml: model_revision must be a 40-character lowercase commit SHA")

    serve = (root / "serve.sh").read_text(encoding="utf-8")
    prepare = (root / "prepare.sh").read_text(encoding="utf-8")
    if assignment(serve, "MODEL_ID") != metadata.get("model"):
        errors.append("serve.sh: MODEL_ID must match submission.yaml model")
    if assignment(serve, "MODEL_REVISION") != revision:
        errors.append("serve.sh: MODEL_REVISION must match submission.yaml model_revision")
    if f'revision="{revision}"' not in prepare:
        errors.append("prepare.sh: snapshot_download revision must match submission.yaml")
    for label, pattern in {
        "served model name": r"--served-model-name\s+submission(?:\s|\\|$)",
        "listen address": r"--host\s+0\.0\.0\.0(?:\s|\\|$)",
        "port": r"--port\s+8000(?:\s|\\|$)",
        "immutable revision argument": r"--revision\s+[\"']?\$\{?MODEL_REVISION\}?[\"']?",
        "4096-token context": r"--max-model-len\s+4096(?:\s|\\|$)",
    }.items():
        if not re.search(pattern, serve):
            errors.append(f"serve.sh: missing {label}")

    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = project.get("project", {}).get("dependencies", [])
    if not any(re.fullmatch(r"vllm==\d+\.\d+\.\d+", item) for item in dependencies):
        errors.append("pyproject.toml: vllm must be pinned with ==MAJOR.MINOR.PATCH")
    lock = (root / "uv.lock").read_text(encoding="utf-8")
    if 'name = "vllm"' not in lock:
        errors.append("uv.lock: vllm package is missing")

    journal = (root / "JOURNAL.md").read_text(encoding="utf-8")
    if not re.search(r"(?m)^## Round \d{2}\b", journal):
        errors.append("JOURNAL.md: add at least one '## Round NN' entry")

    for script in (root / "prepare.sh", root / "serve.sh"):
        if not script.stat().st_mode & stat.S_IXUSR:
            errors.append(f"{script.name}: file must be executable")

    ignored = {"uv.lock"}
    for path in root.rglob("*"):
        if not path.is_file() or ".git" in path.parts or path.name in ignored:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                errors.append(f"{path.relative_to(root)}: possible {label} committed")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--allow-placeholder-author", action="store_true")
    args = parser.parse_args()
    errors = validate(args.root.resolve(), allow_placeholder_author=args.allow_placeholder_author)
    if errors:
        print("\n".join(f"- {error}" for error in errors), file=sys.stderr)
        return 1
    print("submission contract is internally consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
