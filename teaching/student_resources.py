"""Proposed bounded student profile; browser validation remains mandatory.

This does not change the judge or operator gates. Budgets are a starting profile
for sequential 0.5B experiments, not universal package requirements.
"""
from pathlib import Path
import subprocess
from typing import Mapping, Sequence, TypedDict


class StudentResources(TypedDict):
    """Observed headroom, not an allocation guarantee."""

    available_ram_gib: float
    device_used_mib: int
    device_free_mib: int
    profile: str
    idle_required: bool
    active_compute: bool


def assess_student_resources(
    available_ram_bytes: int, gpu_total_mib: int, gpu_used_mib: int,
    active_compute: bool, require_idle: bool = False,
) -> StudentResources:
    """Apply the proposed sequential small-model resource budget.

    Parameters
    ----------
    available_ram_bytes : int
        MemAvailable, in bytes; not total installed RAM or VRAM.
    gpu_total_mib : int
        Total memory of the single selected GPU, in MiB.
    gpu_used_mib : int
        Device-wide used memory, in MiB; not PyTorch allocated memory.
    active_compute : bool
        Whether nvidia-smi lists a compute process on this GPU.
    require_idle : bool, default False
        Startup requires no compute process and 5 GiB host/device headroom.
        During execution retain 2 GiB host and 512 MiB device headroom.

    Returns
    -------
    StudentResources
        Observed RAM GiB, GPU used/free MiB and profile identifier.

    Raises
    ------
    ValueError
        If memory counters are negative or used exceeds total.
    RuntimeError
        If the proposed budget fails or startup would overlap another owner.

    Notes
    -----
    These are provisional stop rules, not certified free-Colab minima, an OS
    sandbox or proof that loading will succeed. Do not kill unknown processes.
    """
    if min(available_ram_bytes, gpu_used_mib) < 0 or gpu_total_mib <= 0 or gpu_used_mib > gpu_total_mib:
        raise ValueError('Invalid observed memory counters')
    ram_gib = available_ram_bytes / 2**30
    free_mib = gpu_total_mib - gpu_used_mib
    host_floor = 5 if require_idle else 2
    gpu_floor = 5120 if require_idle else 512
    observed = f'available host RAM={ram_gib:.2f} GiB; GPU free={free_mib} MiB'
    if require_idle and active_compute:
        raise RuntimeError('STOP: a GPU compute process is active. Stop only your own previous run; '
                           'do not terminate another owner. ' + observed)
    if ram_gib < host_floor or free_mib < gpu_floor:
        raise RuntimeError(f'STOP: student small-model budget requires host >= {host_floor} GiB '
                           f'and device free >= {gpu_floor} MiB. {observed}. Preserve partial '
                           'evidence; inspect your own memory use or use the recorded CPU route '
                           '(not GPU completion). Do not retry the whole experiment blindly.')
    return dict(available_ram_gib=ram_gib, device_used_mib=gpu_used_mib,
                device_free_mib=free_mib, profile='student-small-model-provisional-v1',
                idle_required=require_idle, active_compute=active_compute)


def validate_student_resource_samples(samples: Sequence[Mapping[str, object]]) -> bool:
    """Check the explicit student startup and teardown resource records.

    Parameters
    ----------
    samples : sequence of mappings
        Two idle snapshots in startup/teardown order. RAM is in GiB; GPU
        counters are device-wide MiB, not allocator bytes.

    Returns
    -------
    bool
        True if both records satisfy the provisional profile and recovery.
        This validates recorded fields, not authenticity or free-tier access.

    Raises
    ------
    ValueError
        If the profile, idle observations, counters or recovery disagree.
    """
    if len(samples) != 2:
        raise ValueError('Student service needs startup and teardown samples')
    for sample in samples:
        ram, used, free = (sample.get(k) for k in ('available_ram_gib','device_used_mib','device_free_mib'))
        if (sample.get('profile') != 'student-small-model-provisional-v1'
                or sample.get('idle_required') is not True or sample.get('active_compute') is not False
                or type(ram) not in (int,float) or not 5 <= ram < float('inf')
                or type(used) is not int or used < 0 or type(free) is not int or free < 5120):
            raise ValueError('Invalid provisional student idle resource record')
    if samples[-1]['device_used_mib'] > samples[0]['device_used_mib'] + 64:
        raise ValueError('Student device memory did not recover after teardown')
    return True


def student_resource_gate(require_idle: bool = False) -> StudentResources:
    """Read Linux and nvidia-smi counters without importing GPU libraries.

    Parameters
    ----------
    require_idle : bool, default False
        Apply the startup ownership/headroom checks when True.

    Returns
    -------
    StudentResources
        Current counters and proposed student profile, not a runtime receipt.

    Raises
    ------
    RuntimeError
        If Linux/CUDA counters are unavailable, multiple GPUs are ambiguous,
        startup ownership is unsafe, or the proposed budget fails.
    """
    meminfo = Path('/proc/meminfo')
    if not meminfo.exists():
        raise RuntimeError('Linux memory counters unavailable. Use a Colab GPU runtime for GPU work; '
                           'the CPU learning route remains available.')
    try:
        available = int(next(line.split()[1] for line in meminfo.read_text().splitlines()
                             if line.startswith('MemAvailable:'))) * 1024
        rows = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.total,memory.used',
             '--format=csv,noheader,nounits'], text=True, timeout=10).strip().splitlines()
        if len(rows) != 1:
            raise RuntimeError('This student profile needs one unambiguous GPU, not zero/multiple GPUs.')
        total, used = (int(part.strip()) for part in rows[0].split(','))
        active = bool(subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
            text=True, timeout=10).strip()) if require_idle else False
    except (FileNotFoundError, subprocess.SubprocessError, StopIteration, ValueError) as exc:
        raise RuntimeError('CUDA counters unavailable. In Colab choose a GPU runtime, reconnect '
                           'and inspect its resources; do not call CPU execution GPU evidence.') from exc
    return assess_student_resources(available, total, used, active, require_idle)
