#!/usr/bin/env python3
"""Static, GPU-free gate for student-facing notebooks."""

import ast
import base64
import hashlib
import io
import json
import re
import subprocess
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "seminars").glob("[0-9][0-9]-*.ipynb"))


def code_body(source: str) -> str:
    """Remove only the declared literal-code transport, never skip its body.

    Parameters
    ----------
    source : str
        Exact code cell; a leading %%lsm newline is the only supported wrapper.

    Returns
    -------
    str
        Scientific code for static parsing, not execution.

    Raises
    ------
    ValueError
        Transport arguments, indentation or a duplicate transport are present.
    """
    if source.lstrip().startswith('%%lsm'):
        if not source.startswith('%%lsm\n'):
            raise ValueError('Malformed literal-code transport')
        source = source[len('%%lsm\n'):]
        if source.lstrip().startswith('%%lsm'):
            raise ValueError('Duplicate literal-code transport')
    return source


def check_frozen_runtime(document: dict, label: str) -> None:
    """Verify the exact portable frozen bundle without importing its modules.

    Parameters
    ----------
    document : dict
        Unexecuted student notebook containing provided runtime input cells.
    label : str
        Source name for a diagnostic; no private evidence is needed in a mirror.

    Returns
    -------
    None
        Source/hash checks only, not fresh browser or GPU acceptance.

    Raises
    ------
    ValueError
        Interpreter, bundle integrity, payload membership or isolation drifts.
    """
    metadata=document.get('metadata',{}).get('isolated_runtime',{})
    if metadata.get('managed_python')!='3.11.17':
        raise ValueError(label+': managed Python patch must be pinned')
    by_id={cell['id']:cell for cell in document['cells']}
    inputs=''.join(by_id['lsm-runtime-inputs']['source'])
    tree=ast.parse(inputs)
    payload_node=next(node for node in ast.walk(tree) if isinstance(node,ast.Call)
        and isinstance(node.func,ast.Attribute) and node.func.attr=='b64decode')
    payload=base64.b64decode(ast.literal_eval(payload_node.args[0]),validate=True)
    if hashlib.sha256(payload).hexdigest()!=metadata.get('bundle_sha256'):
        raise ValueError(label+': frozen runtime bundle digest')
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names={'runtime.py','bridge.py','worker.py','pyproject.toml','uv.lock','manifest.json'}
        if set(archive.namelist())!=names:
            raise ValueError(label+': unexpected frozen-runtime member')
        manifest=json.loads(archive.read('manifest.json'))
        if manifest.get('python')!='3.11.17' or manifest.get('dependency_isolation') is not True or manifest.get('os_sandbox') is not False:
            raise ValueError(label+': interpreter/isolation manifest')
        if set(manifest['files'])!=names-{'manifest.json'}:
            raise ValueError(label+': incomplete frozen-runtime file inventory')
        for name,digest in manifest['files'].items():
            if hashlib.sha256(archive.read(name)).hexdigest()!=digest:
                raise ValueError(label+': changed frozen-runtime source '+name)
        installer=archive.read('runtime.py').decode()
        if '08ad3f0a96b536082d6020a29830712c55346dca555f81a337c628f00369f57a' not in installer:
            raise ValueError(label+': managed interpreter integrity absent')
        if '--system-site-packages' in installer:
            raise ValueError(label+': base package inheritance forbidden')
    for identity in ('lsm-runtime-install','lsm-runtime-bridge','lsm-runtime-close'):
        if identity not in by_id:
            raise ValueError(label+': required isolated stage absent: '+identity)
    install=''.join(by_id['lsm-runtime-install']['source'])
    if not all(term in install for term in ("'runtime.py'", "'cpu'", 'RUNTIME_ROOT')):
        raise ValueError(label+': CPU stage must use the supplied managed installer')


def check_numpy_functions(source: str, label: str) -> int:
    """Reject undocumented/untyped function contracts in visible notebook code."""
    tree = ast.parse(code_body(source), filename=label)
    count = 0
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        count += 1
        args = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        args += [a for a in (node.args.vararg, node.args.kwarg) if a is not None]
        args = [a for a in args if a.arg not in ('self', 'cls')]
        if node.returns is None or any(a.annotation is None for a in args):
            raise ValueError(f"{label}: {node.name}: missing type annotation")
        doc = ast.get_docstring(node) or ''
        sections = dict(re.findall(
            r'^([A-Za-z ]+)\n-{3,}\n(.*?)(?=^[A-Za-z ]+\n-{3,}\n|\Z)',
            doc, re.M | re.S))
        if not doc or not (sections.get('Returns') or sections.get('Yields')):
            raise ValueError(f"{label}: {node.name}: missing NumPy Returns/Yields")
        params = sections.get('Parameters', '')
        names = {name.strip().lstrip('*') for group in
                 re.findall(r'^(\*{0,2}\w+(?:,\s*\*{0,2}\w+)*)\s*:\s*\S.*$', params, re.M)
                 for name in group.split(',')}
        if args and (not params or names != {a.arg for a in args}):
            raise ValueError(f"{label}: {node.name}: NumPy Parameters mismatch")
    return count


def check_numpy_fixtures() -> None:
    """Keep missing types, sections and incorrect parameter names rejecting."""
    valid = '''def example(value: int) -> bool:
    """Check a value.

    Parameters
    ----------
    value : int
        Test value.

    Returns
    -------
    bool
        Test outcome.
    """
    return value > 0
'''
    assert check_numpy_functions(valid, 'valid-fixture') == 1
    assert check_numpy_functions(valid.replace('Test value.', 'Gold label: valid prose with a colon.'),'colon-description') == 1
    for changed in (valid.replace('value: int', 'value'),
                    valid.replace(' -> bool', ''),
                    valid.replace('Parameters\n', 'Arguments\n'),
                    valid.replace('Returns\n', 'Result\n'),
                    valid.replace('value : int', 'wrong_name : int')):
        try:
            check_numpy_functions(changed, 'rejecting-fixture')
        except ValueError:
            pass
        else:
            raise AssertionError('Notebook type/docstring regression accepted')


def check_provided_visibility(source: str, metadata: dict, label: str) -> None:
    """Reject giant expanded helper/capture cells that obscure student work.

    Parameters
    ----------
    source : str
        Full visible cell source, including embedded literal modules/data.
    metadata : dict
        Notebook cell display metadata.
    label : str
        Cell identity for a diagnostic.

    Returns
    -------
    None
        Static guard only; the folded display still needs live browser review.
    """
    if len(source) > 10000 and not (code_body(source).startswith('#@title ') and
            metadata.get('cellView') == 'form' and metadata.get('collapsed') is True and
            metadata.get('jupyter', {}).get('source_hidden') is True):
        raise ValueError(f'{label}: giant provided source must be a labelled folded form')


def check_literal_view(source: str) -> None:
    """Reject unwrapped or unescaped Topic01 literal output panels.

    Parameters
    ----------
    source : str
        Joined canonical code-cell sources; raw model evidence is not rewritten.

    Returns
    -------
    None
        A static display contract, not browser or runtime acceptance.

    Raises
    ------
    ValueError
        If one of the required literal views or escaping rules is absent.
    """
    for term in ('white-space:pre-wrap', "show_text('Non-streaming response",
                 "show_text('Streaming response", 'escape(text)'):
        if term not in source:
            raise ValueError('Topic01 literal response viewer must wrap and escape: '+term)


def check_compact_summary(source: str) -> None:
    """Reject wide merged-condition Topic02 summary tables.

    Parameters
    ----------
    source : str
        Joined canonical sources, including selectors and three distinct viewers.

    Returns
    -------
    None
        Requires five-column phase/rate tables and complete raw repeat access.

    Raises
    ------
    ValueError
        Missing selectors, scalar summaries or the compact column inventories.
    """
    for term in ('SUMMARY_MODEL', 'SUMMARY_CASE', 'selected_summary',
                 'tensor_view', '*tensor.items()', "['field','value']",
                 'PROFILE_ROWS', "profile['operators'][:5]",
                 "['input_tokens','operator','calls','self_device_us']",
                 "['phase','unit','median','min','max']",
                 "['metric','unit','median','min','max']",
                 'raw_view', 'row[metric][statistic]', 'assert len(raw_view)==5'):
        if term not in source:
            raise ValueError('Topic02 compact phase/rate/raw-repeat contract missing: '+term)
    if ('show_table(SUMMARY,' in source or 'list(phase_rows[0])' in source
            or 'print(json.dumps(PROFILE,' in source):
        raise ValueError('Topic02 nested or eleven-column summary regression')


def main() -> None:
    check_numpy_fixtures()
    check_provided_visibility('x=1', {}, 'short-fixture')
    check_provided_visibility('#@title Provided fixture\n'+'#'*10001,
        dict(cellView='form',collapsed=True,jupyter=dict(source_hidden=True)), 'folded-fixture')
    try:
        check_provided_visibility('#'*10001, {}, 'rejecting-fixture')
    except ValueError:
        pass
    else:
        raise AssertionError('Expanded giant source regression accepted')
    literal_fixture = "white-space:pre-wrap show_text('Non-streaming response show_text('Streaming response escape(text)"
    check_literal_view(literal_fixture)
    for term in ('white-space:pre-wrap', 'escape(text)'):
        try:
            check_literal_view(literal_fixture.replace(term, 'broken'))
        except ValueError:
            pass
        else:
            raise AssertionError('Unwrapped/unsafe literal viewer regression accepted')
    if len(NOTEBOOKS) < 5:
        raise SystemExit("expected Topics 00–04 notebooks")
    for path in NOTEBOOKS:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("nbformat") != 4 or not data.get("cells"):
            raise ValueError(f"{path.name}: invalid or empty notebook")
        all_source = "".join("".join(cell.get("source", [])) for cell in data["cells"])
        if 'isolated_runtime' in data['metadata']:
            check_frozen_runtime(data,path.name)
        if path.name.startswith('01-'):
            check_literal_view(all_source)
        if path.name.startswith('02-'):
            check_compact_summary(all_source)
            for broken in (all_source.replace("['phase','unit','median','min','max']", 'list(phase_rows[0])'),
                           all_source.replace('tensor_view', 'DENSE_JSON'),
                           all_source.replace('PROFILE_ROWS', 'DENSE_PROFILE_JSON'),
                           all_source.replace('SUMMARY_CASE', 'MISSING_SELECTOR')):
                try:
                    check_compact_summary(broken)
                except ValueError:
                    pass
                else:
                    raise AssertionError('Wide/incomplete summary regression accepted')
        pilot = path.name == "03-quantization-pilot.ipynb"
        if pilot:
            # This independent teaching protocol has student TODOs and an opt-in
            # runner. Do not accidentally impose the archived v3 runner's schema
            # on every file beginning with 03-, or silently skip the new contract.
            # Portable mirror gate: never depend on private course JS modules.
            roles = [c.get("metadata", {}).get("pilot_role") for c in data["cells"]]
            for role in ("student-scale", "student-groups", "real-calibration", "real-held-out", "gpu-optin"):
                if roles.count(role) != 1:
                    raise ValueError(f"{path.name}: missing/duplicate pilot role {role}")
            if not roles.index("real-calibration") < roles.index("real-held-out") < roles.index("gpu-optin"):
                raise ValueError("calibration, held-out, GPU order changed")
            for term in ("def derive_scale(", "def quantize_groups(", "raise NotImplementedError",
                         "REAL_SIZES = [None]", "CHOSEN_GROUP = None", "output_error = None",
                         "RUN_GPU = False", "allow_pickle=False", "4.30", "topic03-layer-decision.json",
                         "CLIP_CANDIDATES = (100, 99, 99.5)", "CHOSEN_CLIP = None",
                         "NF4_RESTORED = nf4_qdq(W_REAL)", "FROZEN_CHOICE = json.dumps"):
                if term not in all_source:
                    raise ValueError(f"{path.name}: student contract missing {term}")
            import hashlib
            for name, key in (("topic03_pilot.py", "runner_sha256"),):
                local = ROOT / "teaching" / name
                if local.exists() and hashlib.sha256(local.read_bytes()).hexdigest() != data["metadata"]["topic03_pilot"][key]:
                    raise ValueError(f"{path.name}: mismatched runner")
        if not pilot and path.name.startswith(("01-", "02-", "03-", "04-")):
            if not all(term in all_source for term in ("LSM_MODEL_CACHE", "snapshot_download", "local_files_only=True")):
                raise ValueError(f"{path.name}: persistent pinned local model cache contract missing")
            if "local_dir=MODEL_DIR" in all_source or not re.search(r"[0-9a-f]{40}", all_source):
                raise ValueError(f"{path.name}: immutable model revision/cache integrity contract missing")
            if 'isolated_runtime' not in data['metadata']:
                install_guard = "if not matches" if path.name.startswith(("03-", "04-")) else "if not installed"
                if not all(term in all_source for term in ("PackageNotFoundError", "version(name)" if not path.name.startswith("04-") else "version('vllm')", install_guard)):
                    raise ValueError(f"{path.name}: pinned package installation must be conditional")
        if path.name.startswith("01-") and not all(
            term in all_source for term in ("worker.join(timeout=30)", "TextIteratorStreamer", "timeout=30")
        ):
            raise ValueError(f"{path.name}: streaming thread needs a bounded timeout")
        if not pilot and path.name.startswith("03-") and not all(
            term in all_source for term in ("topic-03-paired-smoke-v3", "quality_sample", "timing_trials", "min_new_tokens", "natural_eos", "fixed_output_length", "cleanup_verified", "checkpoint(output_path,data)", "validate_artifact", "RECORDED_QUANTIZATION", "toy_namespace")
        ):
            raise ValueError(f"{path.name}: canonical paired v3 clock/quality/checkpoint/CPU contract missing")
        if path.name.startswith("04-") and not all(
            term in all_source for term in ("topic-04-api-smoke-v3", "VLLM_USE_V2_MODEL_RUNNER", "VLLM_USE_FLASHINFER_SAMPLER", "env=env", "normalize_input_ids", "2048<len(rendered)<8192", "ChatStream", "stop_owned_process_group", "max_retries=0", "validate_artifact", "RECORDED_SERVICE", "parser_namespace")
        ):
            raise ValueError(f"{path.name}: canonical v3 WSL/context/byte-replay/owned-teardown contract missing")
        if path.name.startswith("04-"):
            roles = [c.get("metadata", {}).get("investigation_role") for c in data["cells"]]
            for role in ("student-admission", "student-completion", "real-replay", "plots-and-change", "cpu-plan", "gpu-optin", "fresh-service", "freeze-selection", "held-out"):
                if roles.count(role) != 1:
                    raise ValueError(f"{path.name}: missing/duplicate student action {role}")
            if not roles.index("cpu-plan") < roles.index("gpu-optin") < roles.index("fresh-service") < roles.index("freeze-selection") < roles.index("held-out"):
                raise ValueError("Topic04 calibration/freeze/held-out order changed")
            for term in ("def admit_request(", "def score_predictions(", "RUN_GPU=False",
                         "sampling_distribution", "live_task_probe", "SCORING_CHECKS==6",
                         "raise NotImplementedError", "topic-04-client.json", "role_or_empty"):
                if term not in all_source:
                    raise ValueError(f"{path.name}: student investigation missing {term}")
        if path.name.startswith('05-'):
            by_id={c['id']:c for c in data['cells']}
            for identity in ('topic05-account-function','topic05-admission-function'):
                if 'raise NotImplementedError' not in ''.join(by_id[identity]['source']):
                    raise ValueError('Topic05 published student source contains a solved function')
            parameters=''.join(by_id['topic05-gpu-parameters']['source'])
            if 'RUN_GPU=False' not in parameters or 'COHORT_SIZES=[1]' not in parameters:
                raise ValueError('Topic05 GPU opt-in/student intervention defaults drift')
            for identity,variable,name in (
                ('topic05-helper','HELPER_SOURCE','topic05_scheduling.py'),
                ('topic05-gpu-helper','GPU_HELPER_SOURCE','topic05_gpu.py')):
                tree=ast.parse(code_body(''.join(by_id[identity]['source'])))
                assignment=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign)
                    and any(isinstance(t,ast.Name) and t.id==variable for t in n.targets))
                embedded=ast.literal_eval(assignment.value).encode()
                if embedded!=(ROOT/'teaching'/name).read_bytes():
                    raise ValueError('Topic05 embedded/readable helper mismatch: '+name)
            for term in ('LSM_PRIVATE_INSTRUCTOR_NOTES','private-topic05-close'):
                if term in all_source:raise ValueError('Private Topic05 reference leaked')
            if data['metadata']['lsm_topic05'].get('learnerValidation')!='pending':
                raise ValueError('Technical rehearsal cannot certify learner completion')
        code_cells = 0
        cell_ids = set()
        for index, cell in enumerate(data["cells"], 1):
            cell_id = cell.get("id")
            if not isinstance(cell_id, str) or not cell_id or cell_id in cell_ids:
                raise ValueError(f"{path.name} cell {index}: missing or duplicate nbformat cell ID")
            cell_ids.add(cell_id)
            if cell.get("outputs") or cell.get("execution_count") is not None:
                raise ValueError(f"{path.name} cell {index}: published output/execution state")
            source = "".join(cell.get("source", []))
            if cell['cell_type'] == 'code':
                check_provided_visibility(source, cell.get('metadata', {}), f'{path.name}:cell-{index}')
            if "github.com/levshaazz/llm-serving-mastery/" in source:
                raise ValueError(f"{path.name} cell {index}: private source link")
            if cell.get("cell_type") != "code":
                continue
            code_cells += 1
            source = code_body(source)
            if source.lstrip().startswith(("%", "!")):
                continue  # notebook runtime magic/shell cell
            ast.parse(source, filename=f"{path.name}:cell-{index}")
            if path.name.startswith('04-') or 'isolated_runtime' in data['metadata']:
                check_numpy_functions(source, f"{path.name}:cell-{index}")
        if not code_cells:
            raise ValueError(f"{path.name}: no executable code")
        print(f"{path.name}: {len(data['cells'])} cells, {code_cells} code cells passed")
    # Private sources never need to exist in the public student mirror.
    for relative in ('instructor/lecture-notes/topic04/reference.py',
                     'instructor/lecture-notes/topic04/cell-solutions.py',
                     'teaching/topic04_types.py',
                     'teaching/topic03_pilot.py'):
        path = ROOT / relative
        if path.exists():
            count = check_numpy_functions(path.read_text(), relative)
            print(f'{relative}: {count} typed NumPy function contracts passed')


if __name__ == "__main__":
    main()
