#!/usr/bin/env python3
"""Stage-wise LSF Forward Flux Sampling manager.

The runner deliberately submits one independent LSF job per trajectory and
only a single manager process edits campaign_state.json.  This makes the
state pools safe on shared filesystems and lets a later invocation continue
after every completed batch.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable


VERSION = 1
MONITOR_KEYS = {"FFSampling", "PathSampling", "MetaD", "GASMD", "Monitor"}


def _scalar(value: str) -> Any:
    value = value.strip()
    if value.lower() in {"null", "none", "~"}:
        return None
    if value.lower() in {"true", "yes", "on"}:
        return True
    if value.lower() in {"false", "no", "off"}:
        return False
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        return [] if not inner else [_scalar(item) for item in inner.split(",")]
    if value.startswith("{") and value.endswith("}"):
        inner = value[1:-1].strip()
        if not inner:
            return {}
        result: dict[str, Any] = {}
        for item in inner.split(","):
            key, separator, item_value = item.partition(":")
            if not separator:
                raise ValueError(f"Cannot parse inline YAML mapping: {value}")
            result[key.strip()] = _scalar(item_value)
        return result
    try:
        return float(value) if any(mark in value for mark in ".eE") else int(value)
    except ValueError:
        return value.strip("'\"")


def load_yaml(path: Path) -> dict[str, Any]:
    """Read normal YAML, with a small dependency-free fallback for examples."""
    try:
        import yaml  # type: ignore
    except Exception:
        yaml = None
    if yaml is not None:
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, separator, value = line.strip().partition(":")
        if not separator:
            raise ValueError(f"Cannot parse YAML without PyYAML: {raw}")
        while indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip():
            parent[key] = _scalar(value)
        else:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
    return root


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path.resolve()


def resolve_path(base: Path, value: str | Path | None, label: str, required: bool = True) -> Path | None:
    if value is None:
        if required:
            raise ValueError(f"Missing required YAML value: {label}")
        return None
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return require_file(path, label) if required else path.resolve()


def cfg_value(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def normalize_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return shlex.split(value)
    if isinstance(value, list):
        return [str(item) for item in value]
    raise ValueError("Expected a string or list")


def stage_name(kind: str, interface_index: int | None = None) -> str:
    return "flux" if kind == "flux" else f"interface_{interface_index:03d}"


def task_name(label: str, kind: str, interface_index: int | None, attempt: int) -> str:
    interface = "0" if kind == "flux" else str((interface_index or 0) + 1)
    return f"{label}-{interface}-{attempt}"


class Campaign:
    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path.resolve()
        self.base = self.config_path.parent
        self.config = load_yaml(self.config_path)
        self._validate()
        root = cfg_value(self.config, "campaign", "root", default="FFS_remote_runs")
        self.root = (self.base / root).resolve() if not Path(root).is_absolute() else Path(root)
        state_name = cfg_value(self.config, "campaign", "state_file", default="campaign_state.json")
        self.state_path = self.root / str(state_name)

    @property
    def backend(self) -> str:
        return str(cfg_value(self.config, "execution", "backend")).lower()

    @property
    def mode(self) -> str:
        return str(cfg_value(self.config, "execution", "mode", default="gpu")).lower()

    @property
    def interfaces(self) -> list[float]:
        return [float(item) for item in cfg_value(self.config, "ffs", "interfaces")]

    @property
    def direction(self) -> int:
        return 1 if int(cfg_value(self.config, "ffs", "direction", default=1)) >= 0 else -1

    @property
    def randomize_velocities(self) -> bool:
        return bool(cfg_value(self.config, "ffs", "randomize_velocities", default=True))

    def path(self, value: str | Path | None, label: str, required: bool = True) -> Path | None:
        return resolve_path(self.base, value, label, required)

    def _validate(self) -> None:
        interfaces = cfg_value(self.config, "ffs", "interfaces")
        if not isinstance(interfaces, list) or len(interfaces) < 2:
            raise ValueError("ffs.interfaces must contain at least two values")
        if str(cfg_value(self.config, "execution", "backend", default="")).lower() not in {
            "gpu_sampling_ffs", "gpu_sampling_monitor", "lammps_plumed"
        }:
            raise ValueError("execution.backend must be gpu_sampling_ffs, gpu_sampling_monitor, or lammps_plumed")
        if str(cfg_value(self.config, "execution", "mode", default="gpu")).lower() not in {"gpu", "cpu"}:
            raise ValueError("execution.mode must be gpu or cpu")
        if not cfg_value(self.config, "resources", "initial_states"):
            raise ValueError("resources.initial_states must contain one or more equilibrated A-basin structures")

    def initial_states(self) -> list[Path]:
        values = cfg_value(self.config, "resources", "initial_states")
        return [require_file((self.base / value).resolve() if not Path(value).is_absolute() else Path(value), "initial state") for value in values]

    def load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            raise FileNotFoundError(f"Campaign has not been initialized: {self.state_path}")
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def save_state(self, state: dict[str, Any]) -> None:
        state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        write_json(self.state_path, state)

    def init(self) -> dict[str, Any]:
        if self.state_path.exists():
            raise FileExistsError(f"Campaign state already exists: {self.state_path}")
        self.root.mkdir(parents=True, exist_ok=False)
        state = {
            "version": VERSION,
            "config": str(self.config_path),
            "phase": "ready",
            "stage": {"kind": "flux", "interface_index": 0, "round": 0},
            "interfaces": self.interfaces,
            "direction": self.direction,
            "randomize_velocities": self.randomize_velocities,
            "task_counter": 0,
            "state_counter": 0,
            "tasks": {},
            "pools": {str(index): [] for index in range(len(self.interfaces))},
            "statistics": {"flux": {"trials": 0, "successes": 0, "steps": 0}, "interfaces": {}},
            "history": [],
        }
        self.save_state(state)
        return state


def value_crosses(value: float, target: float, fail: float, direction: int) -> int:
    """Return +1 when target precedes failure, -1 on failure, otherwise zero."""
    if direction > 0:
        if value >= target:
            return 1
        if value <= fail:
            return -1
    else:
        if value <= target:
            return 1
        if value >= fail:
            return -1
    return 0


def replace_template(text: str, values: dict[str, Any]) -> str:
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", str(value))
    return text


def _rewrite_velocity(line: str, temperature: float | None, seed: int | None) -> str:
    if seed is None:
        return line
    prefix = line[: len(line) - len(line.lstrip())]
    command, marker, comment = line.strip().partition("#")
    parts = command.split()
    if not parts or parts[0] != "velocity":
        return line
    if temperature is not None and len(parts) >= 2:
        parts[1] = f"{temperature:g}"
    if len(parts) == 2:
        parts.extend(["seed", str(seed)])
    elif len(parts) >= 4 and parts[2] == "seed":
        parts[3] = str(seed)
    else:
        raise ValueError("A randomized GPU-Sampling template must use 'velocity T' or 'velocity T seed N'.")
    rendered = prefix + " ".join(parts)
    return rendered + ((" #" + comment) if marker else "")


def rewrite_gpu_run(template: Path, *, monitor_line: str, steps: int, nep_name: str, random_seed: int | None,
                    temperature: float | None, monitor_dump_interval: int | None) -> str:
    lines: list[str] = []
    inserted = False
    saw_velocity = False
    for raw in template.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            lines.append(raw)
            continue
        key = stripped.split()[0]
        if key in MONITOR_KEYS or key == "run" or (monitor_dump_interval is not None and key == "dump_exyz"):
            continue
        if key == "potential":
            lines.append("potential " + nep_name)
            continue
        if key == "velocity":
            saw_velocity = True
            lines.append(_rewrite_velocity(raw, temperature, random_seed))
            continue
        lines.append(raw)
    if random_seed is not None and not saw_velocity:
        if temperature is None:
            raise ValueError("md.temperature_K is required when randomize_velocities is true and run.in has no velocity line")
        lines.append(f"velocity {temperature:g} seed {random_seed}")
    lines.append(monitor_line)
    if monitor_dump_interval is not None:
        lines.append(f"dump_exyz {monitor_dump_interval} 1")
    lines.append(f"run {steps}")
    return "\n".join(lines).rstrip() + "\n"


def append_kv_overrides(source: Path, target: Path, values: dict[str, Any]) -> None:
    text = source.read_text(encoding="utf-8").rstrip() + "\n\n# FFS remote-runner overrides\n"
    for key, value in values.items():
        if isinstance(value, bool):
            value = "true" if value else "false"
        text += f"{key}: {value}\n"
    target.write_text(text, encoding="utf-8")


def read_xyz_atoms(path: Path) -> int:
    try:
        return int(path.read_text(encoding="utf-8", errors="replace").splitlines()[0].strip())
    except (IndexError, ValueError) as error:
        raise ValueError(f"Cannot infer atom count from XYZ file: {path}") from error


def strip_extxyz_velocities(path: Path) -> bool:
    """Remove extxyz vel/velocity columns before assigning a fresh seed."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 2:
        return False
    match = re.search(r"Properties=(['\"])(.*?)\1", lines[1])
    if not match:
        return False
    fields = match.group(2).split(":")
    if len(fields) % 3:
        raise ValueError(f"Malformed extxyz Properties declaration: {path}")
    offset = 0
    kept: list[str] = []
    velocity_offset = velocity_count = None
    for index in range(0, len(fields), 3):
        name, kind, count_text = fields[index:index + 3]
        count = int(count_text)
        if name.lower() in {"vel", "velocity", "velocities"}:
            velocity_offset, velocity_count = offset, count
        else:
            kept.extend([name, kind, count_text])
        offset += count
    if velocity_offset is None or velocity_count is None:
        return False
    atoms = read_xyz_atoms(path)
    if len(lines) < atoms + 2:
        raise ValueError(f"Truncated extxyz file: {path}")
    original = match.group(0)
    replacement = f"Properties={match.group(1)}{':'.join(kept)}{match.group(1)}"
    lines[1] = lines[1].replace(original, replacement, 1)
    for index in range(2, atoms + 2):
        row = lines[index].split()
        if len(row) < offset:
            raise ValueError(f"Atom line has fewer columns than Properties in {path}")
        del row[velocity_offset:velocity_offset + velocity_count]
        lines[index] = " ".join(row)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


def copy_resource(source: Path, task_dir: Path) -> Path:
    target = task_dir / source.name
    shutil.copy2(source, target)
    return target


def parse_lsf_template(template: Path, *, queue: str, job_name: str, mode: str,
                       gpu_request: str, cpu_cores: int) -> list[str]:
    lines: list[str] = []
    queue_done = job_done = resource_done = False
    for raw in template.read_text(encoding="utf-8").splitlines():
        stripped = raw.lstrip()
        # An exclamation-marked #BSUB is an LSF-specific disabled directive;
        # preserve it byte-for-byte and never use it as an active control line.
        if stripped.startswith("! #BSUB"):
            lines.append(raw)
            continue
        if stripped.startswith("#BSUB -q"):
            if not queue_done:
                lines.append(f"#BSUB -q {queue}")
                queue_done = True
            continue
        if stripped.startswith("#BSUB -J"):
            if not job_done:
                lines.append(f"#BSUB -J {job_name}")
                job_done = True
            continue
        if stripped.startswith("#BSUB -gpu") or stripped.startswith("#BSUB -n"):
            if not resource_done:
                lines.append(f'#BSUB -gpu "{gpu_request}"' if mode == "gpu" else f"#BSUB -n {cpu_cores}")
                resource_done = True
            continue
        lines.append(raw)
    insert_at = 1 if lines and lines[0].startswith("#!") else 0
    missing: list[str] = []
    if not queue_done:
        missing.append(f"#BSUB -q {queue}")
    if not job_done:
        missing.append(f"#BSUB -J {job_name}")
    if not resource_done:
        missing.append(f'#BSUB -gpu "{gpu_request}"' if mode == "gpu" else f"#BSUB -n {cpu_cores}")
    return lines[:insert_at] + missing + lines[insert_at:]


def render_job(template: Path, *, queue: str, job_name: str, mode: str, gpu_request: str,
               cpu_cores: int, command: list[str], backend: str) -> str:
    lines = parse_lsf_template(template, queue=queue, job_name=job_name, mode=mode,
                               gpu_request=gpu_request, cpu_cores=cpu_cores)
    filtered: list[str] = []
    auto_pattern = re.compile(r"\bgpu-sampling(?:\s|$)") if backend.startswith("gpu_") else None
    for raw in lines:
        if "{{FFS_COMMAND}}" in raw:
            continue
        if auto_pattern is not None and not raw.lstrip().startswith(("#", "!")) and auto_pattern.search(raw):
            continue
        filtered.append(raw)
    command_line = " ".join(shlex.quote(token) for token in command)
    filtered.extend([
        "",
        "# FFS remote-runner command; do not replace this generated block.",
        "set +e",
        command_line + " > ffs_runtime.out 2>&1",
        "FFS_REMOTE_RC=$?",
        "printf '%s\\n' \"$FFS_REMOTE_RC\" > .ffs_exit_code",
        "exit \"$FFS_REMOTE_RC\"",
        "",
    ])
    return "\n".join(filtered)


def sanitize_lsf_for_local_shell(text: str) -> str:
    """Turn LSF metadata into shell comments for simulation syntax checks."""
    output: list[str] = []
    for raw in text.splitlines():
        stripped = raw.lstrip()
        if stripped.startswith("#BSUB") or stripped.startswith("! #BSUB"):
            output.append("# " + raw)
        else:
            output.append(raw)
    return "\n".join(output) + "\n"


def validate_simulated_shell(text: str) -> None:
    """Use bash syntax checking when accessible, otherwise verify runner invariants.

    Windows can expose bash.exe while the WSL service is unavailable to a
    sandboxed Python subprocess.  The structural checks preserve deterministic
    simulation coverage in that case; an accessible bash still checks syntax.
    """
    if "{{FFS_" in text:
        raise RuntimeError("Generated job still contains an unresolved FFS placeholder")
    if "FFS_REMOTE_RC=$?" not in text or ".ffs_exit_code" not in text:
        raise RuntimeError("Generated job is missing the remote-runner exit-code block")
    bash = shutil.which("bash")
    if bash:
        check_command = [bash, "-n", "-"]
    elif shutil.which("wsl"):
        check_command = ["wsl", "-d", "Ubuntu", "--", "bash", "-n", "-"]
    else:
        return
    completed = subprocess.run(check_command, input=text, text=True, capture_output=True, check=False)
    if completed.returncode:
        combined = completed.stdout + completed.stderr
        normalized = combined.replace("\x00", "")
        if "CreateInstance/E_ACCESSDENIED" not in normalized:
            raise RuntimeError(f"Generated job is not shell-valid: {combined}")


def parse_numeric_log(path: Path) -> tuple[list[str], list[list[float]]]:
    headers: list[str] = []
    rows: list[list[float]] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = raw.strip()
        if stripped.startswith("#!") and "FIELDS" in stripped:
            headers = stripped.split("FIELDS", 1)[1].split()
            continue
        if not stripped or stripped.startswith("#"):
            continue
        try:
            rows.append([float(item) for item in stripped.split()])
        except ValueError:
            continue
    return headers, rows


def select_cv_column(headers: list[str], value: str | int | None) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    if isinstance(value, str):
        if value not in headers:
            raise ValueError(f"CV column {value!r} is absent from log header {headers}")
        return headers.index(value)
    return 0


def classify_cv_log(path: Path, *, cv_column: str | int | None, target: float, fail: float | None,
                    direction: int) -> tuple[int, int, float | None, float | None]:
    headers, rows = parse_numeric_log(path)
    column = select_cv_column(headers, cv_column)
    for index, row in enumerate(rows):
        if column >= len(row):
            raise ValueError(f"CV column {column} is outside row in {path}")
        if direction > 0:
            status = 1 if row[column] >= target else (-1 if fail is not None and row[column] <= fail else 0)
        else:
            status = 1 if row[column] <= target else (-1 if fail is not None and row[column] >= fail else 0)
        if status:
            return status, index, row[column], row[0] if row else None
    return 0, len(rows) - 1, rows[-1][column] if rows else None, rows[-1][0] if rows else None


def classify_gpu_ffs(path: Path) -> tuple[int, int, float | None]:
    _, rows = parse_numeric_log(path)
    if not rows:
        return 0, -1, None
    final = rows[-1]
    status = int(round(final[0]))
    return (1 if status > 0 else -1 if status < 0 else 0), len(rows) - 1, final[1] if len(final) > 1 else None


def final_cv_log_value(path: Path, cv_column: str | int | None) -> tuple[int, float | None, float | None]:
    """Return the final measured CV row without applying any interface test."""
    headers, rows = parse_numeric_log(path)
    if not rows:
        return -1, None, None
    column = select_cv_column(headers, cv_column)
    if column >= len(rows[-1]):
        raise ValueError(f"CV column {column} is outside final row in {path}")
    return len(rows) - 1, rows[-1][column], rows[-1][0] if rows[-1] else None


def local_crossing_decision(campaign: Campaign, task: dict[str, Any], seed: int,
                            success_probability: float) -> dict[str, Any]:
    """Create a reproducible test-only crossing decision next to real outputs."""
    if campaign.backend == "gpu_sampling_ffs":
        raise ValueError("run-local --random-crossings requires a monitor backend; "
                         "gpu_sampling_ffs must use its native boundary decision")
    if not 0.0 <= success_probability <= 1.0:
        raise ValueError("--success-probability must be between 0 and 1")
    digest = hashlib.sha256(f"{seed}:{task['id']}".encode("utf-8")).digest()
    draw = int.from_bytes(digest[:8], "big") / float(1 << 64)
    decision = {
        "kind": "local_random_crossing",
        "seed": seed,
        "draw": draw,
        "success_probability": success_probability,
        "outcome": "success" if draw < success_probability else "failure",
    }
    write_json(Path(task["work_dir"]) / ".ffs_local_decision.json", decision)
    return decision


def read_local_crossing_decision(task_dir: Path) -> dict[str, Any] | None:
    path = task_dir / ".ffs_local_decision.json"
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("kind") != "local_random_crossing" or value.get("outcome") not in {"success", "failure"}:
        raise ValueError(f"Invalid local random-crossing decision: {path}")
    return value


def read_extxyz_frames(path: Path) -> list[list[str]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    frames: list[list[str]] = []
    cursor = 0
    while cursor < len(lines):
        if not lines[cursor].strip():
            cursor += 1
            continue
        try:
            atoms = int(lines[cursor].strip())
        except ValueError as error:
            raise ValueError(f"Malformed extended XYZ frame in {path} at line {cursor + 1}") from error
        end = cursor + atoms + 2
        if end > len(lines):
            raise ValueError(f"Truncated extended XYZ frame in {path}")
        frames.append(lines[cursor:end])
        cursor = end
    return frames


def read_lammps_dump_frames(path: Path) -> list[tuple[int, list[str]]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    frames: list[tuple[int, list[str]]] = []
    cursor = 0
    while cursor < len(lines):
        if lines[cursor].strip() != "ITEM: TIMESTEP":
            cursor += 1
            continue
        if cursor + 8 >= len(lines):
            raise ValueError(f"Truncated LAMMPS dump frame in {path}")
        step = int(lines[cursor + 1].strip())
        try:
            atom_marker = lines.index("ITEM: NUMBER OF ATOMS", cursor, min(len(lines), cursor + 8))
            atoms = int(lines[atom_marker + 1].strip())
            atoms_header = next(index for index in range(atom_marker + 2, len(lines)) if lines[index].startswith("ITEM: ATOMS"))
        except (ValueError, StopIteration, IndexError) as error:
            raise ValueError(f"Malformed LAMMPS dump frame in {path}") from error
        end = atoms_header + atoms + 1
        if end > len(lines):
            raise ValueError(f"Truncated atom data in {path}")
        frame = list(lines[cursor:end])
        frame[1] = "0"
        frames.append((step, frame))
        cursor = end
    return frames


def deterministic_seed(campaign: Campaign, task_id: str) -> int:
    base_seed = int(cfg_value(campaign.config, "ffs", "random_seed", default=20260808))
    digest = hashlib.sha256((str(base_seed) + ":" + task_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % 2147483646 + 1


def stage_limits(campaign: Campaign, kind: str, interface_index: int) -> tuple[int, int, int, int]:
    if kind == "flux":
        cfg = cfg_value(campaign.config, "ffs", "flux", default={})
        return (int(cfg.get("batch_size", 1)), int(cfg.get("min_successes", 1)),
                int(cfg.get("max_trials", cfg.get("batch_size", 1))), int(cfg.get("steps", 10000)))
    cfg = cfg_value(campaign.config, "ffs", "interface", default={})
    steps = cfg.get("steps", 10000)
    if isinstance(steps, list):
        steps = steps[interface_index]
    return (int(cfg.get("batch_size", 1)), int(cfg.get("min_successes", 1)),
            int(cfg.get("max_trials", cfg.get("batch_size", 1))), int(steps))


def copy_extras(campaign: Campaign, task_dir: Path) -> None:
    for raw in cfg_value(campaign.config, "resources", "extra_files", default=[]):
        copy_resource(require_file((campaign.base / raw).resolve() if not Path(raw).is_absolute() else Path(raw), "extra file"), task_dir)


def render_lammps_input(campaign: Campaign, task: dict[str, Any], task_dir: Path, start_is_initial: bool) -> None:
    lammps = cfg_value(campaign.config, "lammps", default={})
    resources = cfg_value(campaign.config, "resources", default={})
    template = require_file(campaign.path(lammps.get("input_template"), "lammps.input_template"), "LAMMPS input template")
    plumed_template = require_file(campaign.path(lammps.get("plumed_template"), "lammps.plumed_template"), "PLUMED template")
    base_data = require_file(campaign.path(lammps.get("base_data"), "lammps.base_data"), "LAMMPS base data")
    base_target = copy_resource(base_data, task_dir)
    nep = require_file(campaign.path(resources.get("nep"), "resources.nep"), "NEP potential")
    nep_target = copy_resource(nep, task_dir)
    if not start_is_initial:
        shutil.copy2(Path(task["start_path"]), task_dir / "start.lammpstrj")
        start_command = f"read_data {base_target.name}\nread_dump start.lammpstrj 0 x y z vx vy vz box yes replace yes"
    else:
        initial_source = Path(task["start_path"]).resolve()
        initial_target = base_target if initial_source == base_data.resolve() else copy_resource(initial_source, task_dir)
        start_command = f"read_data {initial_target.name}"
    cv_path = str(lammps.get("cv_log", "COLVAR"))
    stride = int(lammps.get("monitor_stride", 1))
    plumed_text = replace_template(plumed_template.read_text(encoding="utf-8"), {
        "FFS_CV_LOG": cv_path,
        "FFS_MONITOR_STRIDE": stride,
    })
    (task_dir / "plumed.dat").write_text(plumed_text, encoding="utf-8")
    trajectory = str(lammps.get("trajectory", "ffs-trajectory.lammpstrj"))
    values = {
        "FFS_START_COMMAND": start_command,
        "FFS_PLUMED_FILE": "plumed.dat",
        "FFS_NEP_FILE": nep_target.name,
        "FFS_STEPS": task["steps"],
        "FFS_CV_LOG": cv_path,
        "FFS_MONITOR_STRIDE": stride,
        "FFS_TRAJECTORY_FILE": trajectory,
        "FFS_TRAJECTORY_COMMAND": f"dump ffs all custom {stride} {trajectory} id type x y z vx vy vz",
    }
    text = replace_template(template.read_text(encoding="utf-8"), values)
    lines: list[str] = []
    saw_start = saw_run = saw_plumed = saw_dump = False
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            lines.append(raw)
            continue
        if stripped.startswith("read_data"):
            if not saw_start:
                lines.extend(start_command.splitlines())
                saw_start = True
            continue
        if stripped.startswith("run "):
            if not saw_dump:
                lines.append(values["FFS_TRAJECTORY_COMMAND"])
                saw_dump = True
            lines.append(f"run {task['steps']}")
            saw_run = True
            continue
        if " plumed " in (" " + stripped + " "):
            lines.append(re.sub(r"plumedfile\s+\S+", "plumedfile plumed.dat", raw))
            saw_plumed = True
            continue
        if stripped.startswith("dump ") and trajectory in stripped:
            saw_dump = True
        lines.append(raw)
    if not saw_start:
        lines.insert(0, start_command)
    if not saw_plumed:
        raise ValueError("LAMMPS input needs a PLUMED fix line or a {{FFS_PLUMED_FILE}} placeholder")
    if not saw_run:
        if not saw_dump:
            lines.append(values["FFS_TRAJECTORY_COMMAND"])
        lines.append(f"run {task['steps']}")
    (task_dir / "in.lammps").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def prepare_task(campaign: Campaign, state: dict[str, Any], task: dict[str, Any]) -> None:
    task_dir = Path(task["work_dir"])
    task_dir.mkdir(parents=True, exist_ok=False)
    resources = cfg_value(campaign.config, "resources", default={})
    start_is_initial = task.get("start_kind") == "initial"
    if campaign.backend.startswith("gpu_sampling"):
        source = Path(task["start_path"])
        shutil.copy2(source, task_dir / "model.xyz")
        if campaign.randomize_velocities:
            strip_extxyz_velocities(task_dir / "model.xyz")
        nep = copy_resource(require_file(campaign.path(resources.get("nep"), "resources.nep"), "NEP potential"), task_dir)
        pt = copy_resource(require_file(campaign.path(resources.get("pt"), "resources.pt"), "TorchScript CV model"), task_dir)
        cfg_source = require_file(campaign.path(resources.get("gas_config"), "resources.gas_config"), "GAS config")
        temperature = cfg_value(campaign.config, "md", "temperature_K", default=None)
        if campaign.backend == "gpu_sampling_ffs":
            runtime_cfg = task_dir / "GAScfg.ffs.yaml"
            append_kv_overrides(cfg_source, runtime_cfg, {
                "n_atoms": read_xyz_atoms(task_dir / "model.xyz"),
                "ffs_enabled": True,
                "ffs_stop_on_fail": task["kind"] != "flux",
                "ffs_cv_index": int(cfg_value(campaign.config, "monitor", "cv_index", default=0)),
                "ffs_direction": campaign.direction,
                "ffs_target_interface": task["target"],
                "ffs_fail_interface": task["fail"],
            })
            monitor_line = f"FFSampling {pt.name} {runtime_cfg.name}"
            monitor_dump = None
            task["cv_output"] = "GASCVlog.txt"
        else:
            runtime_cfg = task_dir / "CVMonitor.yaml"
            cv_output = str(cfg_value(campaign.config, "monitor", "output", default="MonCVlog.txt"))
            append_kv_overrides(cfg_source, runtime_cfg, {"cv_log_path": cv_output})
            monitor_line = f"Monitor {pt.name} {runtime_cfg.name}"
            monitor_dump = int(cfg_value(campaign.config, "monitor", "stride", default=1))
            task["cv_output"] = cv_output
            task["trajectory"] = "dump.xyz"
        template = require_file(campaign.path(cfg_value(campaign.config, "execution", "run_template"), "execution.run_template"), "run.in template")
        (task_dir / "run.in").write_text(rewrite_gpu_run(
            template, monitor_line=monitor_line, steps=task["steps"], nep_name=nep.name,
            random_seed=task.get("velocity_seed") if campaign.randomize_velocities else None,
            temperature=float(temperature) if temperature is not None else None,
            monitor_dump_interval=monitor_dump,
        ), encoding="utf-8")
    else:
        render_lammps_input(campaign, task, task_dir, start_is_initial)
        task["cv_output"] = str(cfg_value(campaign.config, "lammps", "cv_log", default="COLVAR"))
        task["trajectory"] = str(cfg_value(campaign.config, "lammps", "trajectory", default="ffs-trajectory.lammpstrj"))
    copy_extras(campaign, task_dir)
    command_prefix = normalize_list(cfg_value(campaign.config, "execution", "command_prefix", default=[]))
    executable = str(require_file(campaign.path(cfg_value(campaign.config, "execution", "executable"), "execution.executable"), "executable"))
    command = command_prefix + [executable]
    command += ["-m", "model.xyz", "-i", "run.in"] if campaign.backend.startswith("gpu_sampling") else ["-in", "in.lammps"]
    scheduler = cfg_value(campaign.config, "scheduler", default={})
    template = require_file(campaign.path(scheduler.get("job_template"), "scheduler.job_template"), "LSF job template")
    job = render_job(
        template,
        queue=str(scheduler.get("queue", "normal")),
        job_name=task["job_name"], mode=campaign.mode,
        gpu_request=str(scheduler.get("gpu_request", "num=1:aff=yes")),
        cpu_cores=int(scheduler.get("cpu_cores", 24)), command=command, backend=campaign.backend,
    )
    (task_dir / "job.lsf").write_text(job, encoding="utf-8")
    write_json(task_dir / "task.json", task)


def available_starts(state: dict[str, Any], interface_index: int, randomize: bool) -> list[dict[str, Any]]:
    pool = state["pools"][str(interface_index)]
    if randomize:
        return pool
    return [item for item in pool if not item.get("consumed_by")]


def plan(campaign: Campaign) -> list[dict[str, Any]]:
    state = campaign.load_state()
    if state["phase"] in {"complete", "failed"}:
        return []
    outstanding = [item for item in state["tasks"].values() if item["status"] in {"planned", "submitted"}]
    if outstanding:
        raise RuntimeError("Cannot plan a new batch while prior tasks are planned or submitted")
    stage = state["stage"]
    kind = stage["kind"]
    interface_index = int(stage["interface_index"])
    batch_size, _, max_trials, steps = stage_limits(campaign, kind, interface_index)
    key = "flux" if kind == "flux" else str(interface_index)
    statistics = state["statistics"]["flux"] if kind == "flux" else state["statistics"]["interfaces"].setdefault(key, {"trials": 0, "successes": 0})
    remaining = max_trials - int(statistics["trials"])
    if remaining <= 0:
        state["phase"] = "failed"
        state["history"].append({"event": "max_trials_reached", "stage": stage_name(kind, interface_index)})
        campaign.save_state(state)
        return []
    if kind == "flux":
        starts = [{"path": str(item), "kind": "initial"} for item in campaign.initial_states()]
    else:
        starts = [{"path": item["path"], "kind": "state", "state": item} for item in available_starts(state, interface_index, campaign.randomize_velocities)]
        if not starts:
            state["phase"] = "failed"
            state["history"].append({"event": "no_available_start_states", "stage": stage_name(kind, interface_index)})
            campaign.save_state(state)
            return []
    tasks: list[dict[str, Any]] = []
    replicas = int(cfg_value(campaign.config, "ffs", "replicas_per_crossing", default=1)) if campaign.randomize_velocities else 1
    target = campaign.interfaces[0] if kind == "flux" else campaign.interfaces[interface_index + 1]
    fail = float(cfg_value(campaign.config, "ffs", "fail_interface", default=campaign.interfaces[0]))
    label = str(cfg_value(campaign.config, "campaign", "name", default="FFS"))
    candidates: list[dict[str, Any]] = []
    for start in starts:
        for _ in range(replicas):
            candidates.append(start)
    candidates = candidates[: min(batch_size, remaining)]
    for start in candidates:
        state["task_counter"] += 1
        attempt = state["task_counter"]
        task_id = f"{stage_name(kind, interface_index)}-{attempt:06d}"
        work_dir = campaign.root / "tasks" / stage_name(kind, interface_index) / f"attempt_{attempt:06d}"
        task = {
            "id": task_id,
            "kind": kind,
            "interface_index": interface_index,
            "attempt": attempt,
            "job_name": task_name(label, kind, interface_index, attempt),
            "work_dir": str(work_dir),
            "start_path": start["path"],
            "start_kind": start["kind"],
            "start_state_id": start.get("state", {}).get("id"),
            "target": target,
            "fail": fail,
            "steps": steps,
            "velocity_seed": deterministic_seed(campaign, task_id) if campaign.randomize_velocities else None,
            "status": "planned",
        }
        if not campaign.randomize_velocities and start.get("state") is not None:
            start["state"]["consumed_by"] = task_id
        prepare_task(campaign, state, task)
        state["tasks"][task_id] = task
        tasks.append(task)
    state["phase"] = "running"
    state["history"].append({"event": "planned", "stage": stage_name(kind, interface_index), "tasks": [task["id"] for task in tasks]})
    campaign.save_state(state)
    return tasks


def submit(campaign: Campaign) -> list[str]:
    state = campaign.load_state()
    submitted: list[str] = []
    for task in state["tasks"].values():
        if task["status"] != "planned":
            continue
        job = Path(task["work_dir"]) / "job.lsf"
        completed = subprocess.run(["bsub"], input=job.read_text(encoding="utf-8"), text=True,
                                   cwd=job.parent, capture_output=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"bsub failed for {task['id']}: {completed.stdout}{completed.stderr}")
        match = re.search(r"<([0-9]+)>", completed.stdout + completed.stderr)
        if not match:
            raise RuntimeError(f"Could not parse LSF job id for {task['id']}: {completed.stdout}{completed.stderr}")
        task["status"] = "submitted"
        task["lsf_job_id"] = match.group(1)
        submitted.append(task["id"])
    if submitted:
        state["history"].append({"event": "submitted", "tasks": submitted})
        campaign.save_state(state)
    return submitted


def task_exit_code(task_dir: Path) -> int | None:
    marker = task_dir / ".ffs_exit_code"
    if not marker.exists():
        return None
    try:
        return int(marker.read_text(encoding="utf-8").strip())
    except ValueError:
        return 1


def refresh_lsf_status(tasks: Iterable[dict[str, Any]]) -> dict[str, str]:
    """Best-effort LSF state lookup for jobs that did not write completion markers."""
    executable = shutil.which("bjobs")
    job_ids = [str(task["lsf_job_id"]) for task in tasks if task.get("lsf_job_id")]
    if not executable or not job_ids:
        return {}
    completed = subprocess.run([executable, "-noheader", *job_ids], text=True, capture_output=True, check=False)
    if completed.returncode:
        return {}
    states: dict[str, str] = {}
    for raw in completed.stdout.splitlines():
        parts = raw.split()
        if len(parts) >= 3 and parts[0].split("[")[0] in job_ids:
            states[parts[0].split("[")[0]] = parts[2]
    return states


def make_state_record(campaign: Campaign, state: dict[str, Any], task: dict[str, Any], source: Path,
                      suffix: str, details: dict[str, Any]) -> dict[str, Any]:
    destination_index = 0 if task["kind"] == "flux" else int(task["interface_index"]) + 1
    state["state_counter"] += 1
    state_id = f"state_{destination_index:03d}_{state['state_counter']:06d}"
    destination = campaign.root / "states" / f"interface_{destination_index:03d}"
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / (state_id + suffix)
    shutil.copy2(source, output)
    record = {"id": state_id, "path": str(output), "origin_task": task["id"], **details}
    state["pools"][str(destination_index)].append(record)
    return record


def extract_gpu_monitor_state(task_dir: Path, event_row: int, start_path: Path) -> Path:
    if event_row == 0:
        return start_path
    frames = read_extxyz_frames(task_dir / "dump.xyz")
    frame_index = event_row - 1
    if frame_index < 0 or frame_index >= len(frames):
        raise ValueError("CVOnlyMonitor event cannot be aligned to dump.xyz; set monitor.stride to match dump_exyz")
    output = task_dir / "crossing.xyz"
    output.write_text("\n".join(frames[frame_index]) + "\n", encoding="utf-8")
    return output


def extract_lammps_state(campaign: Campaign, task_dir: Path, event_row: int, event_time: float | None) -> Path:
    lammps = cfg_value(campaign.config, "lammps", default={})
    frames = read_lammps_dump_frames(task_dir / str(lammps.get("trajectory", "ffs-trajectory.lammpstrj")))
    if not frames:
        raise ValueError("No LAMMPS trajectory frames were written for a successful CV crossing")
    selected = None
    if event_time is not None and lammps.get("time_to_step") is not None:
        expected = round(float(event_time) * float(lammps["time_to_step"]))
        selected = next((frame for step, frame in frames if step == expected), None)
    if selected is None:
        # Both PLUMED and GPU-Sampling monitor the initial state before the
        # first trajectory dump.  Therefore row k>0 maps to frame k-1 when
        # exactly k frames have been written.
        if event_row > 0 and len(frames) == event_row:
            selected = frames[event_row - 1][1]
        elif len(frames) == event_row + 1:
            selected = frames[event_row][1]
        elif bool(lammps.get("allow_index_matching", False)) and event_row < len(frames):
            selected = frames[event_row][1]
        else:
            raise ValueError("Cannot align first COLVAR crossing to LAMMPS dump; use matching strides or set lammps.allow_index_matching")
    output = task_dir / "crossing.lammpstrj"
    output.write_text("\n".join(selected) + "\n", encoding="utf-8")
    return output


def collect(campaign: Campaign) -> dict[str, int]:
    state = campaign.load_state()
    counts = {"collected": 0, "success": 0, "failure": 0, "incomplete": 0, "runtime_failed": 0, "waiting": 0}
    pending = [task for task in state["tasks"].values() if task["status"] in {"planned", "submitted"} and task_exit_code(Path(task["work_dir"])) is None]
    lsf_states = refresh_lsf_status(pending)
    for task in state["tasks"].values():
        if task["status"] not in {"planned", "submitted"}:
            continue
        task_dir = Path(task["work_dir"])
        code = task_exit_code(task_dir)
        if code is None:
            lsf_status = lsf_states.get(str(task.get("lsf_job_id", "")))
            if lsf_status:
                task["lsf_status"] = lsf_status
            if lsf_status in {"EXIT", "ZOMBI", "UNKWN"}:
                task["status"] = "runtime_failed"
                task["return_code"] = None
                counts["runtime_failed"] += 1
                continue
            counts["waiting"] += 1
            continue
        counts["collected"] += 1
        task["return_code"] = code
        if code != 0:
            task["status"] = "runtime_failed"
            counts["runtime_failed"] += 1
            continue
        decision = read_local_crossing_decision(task_dir)
        if campaign.backend == "gpu_sampling_ffs":
            outcome, event_row, value = classify_gpu_ffs(task_dir / task["cv_output"])
            event_time = None
        else:
            column = cfg_value(campaign.config, "monitor", "cv_index", default=0) if campaign.backend == "gpu_sampling_monitor" else cfg_value(campaign.config, "lammps", "cv_column", default=None)
            outcome, event_row, value, event_time = classify_cv_log(
                task_dir / task["cv_output"], cv_column=column,
                target=float(task["target"]),
                fail=None if task["kind"] == "flux" else float(task["fail"]), direction=campaign.direction)
            if decision is not None:
                # The MD calculation and its unmodified CV/trajectory outputs are
                # still required.  Only the testing classification is overridden.
                event_row, value, event_time = final_cv_log_value(task_dir / task["cv_output"], column)
                outcome = 1 if decision["outcome"] == "success" else -1
                task["local_test_decision"] = decision
        task["cv_value"] = value
        task["event_row"] = event_row
        stage_stats = state["statistics"]["flux"] if task["kind"] == "flux" else state["statistics"]["interfaces"].setdefault(str(task["interface_index"]), {"trials": 0, "successes": 0})
        stage_stats["trials"] += 1
        if task["kind"] == "flux":
            stage_stats["steps"] += int(task["steps"])
        if outcome > 0:
            if campaign.backend == "gpu_sampling_ffs":
                source = task_dir / "restart.xyz"
            elif campaign.backend == "gpu_sampling_monitor":
                source = extract_gpu_monitor_state(task_dir, event_row, Path(task["start_path"]))
            else:
                source = extract_lammps_state(campaign, task_dir, event_row, event_time)
            if not source.exists():
                raise FileNotFoundError(f"Successful task did not create a crossing structure: {source}")
            task["state"] = make_state_record(campaign, state, task, source,
                                               ".xyz" if campaign.backend.startswith("gpu_sampling") else ".lammpstrj",
                                               {"cv": value, "event_row": event_row})
            stage_stats["successes"] += 1
            task["status"] = "success"
            counts["success"] += 1
        elif outcome < 0:
            task["status"] = "failure"
            counts["failure"] += 1
        else:
            task["status"] = "incomplete"
            counts["incomplete"] += 1
    if counts["collected"]:
        state["history"].append({"event": "collected", **counts})
        campaign.save_state(state)
    return counts


def advance(campaign: Campaign) -> str:
    state = campaign.load_state()
    if any(item["status"] in {"planned", "submitted"} for item in state["tasks"].values()):
        return "waiting"
    if any(item["status"] == "runtime_failed" for item in state["tasks"].values()):
        state["phase"] = "failed"
        state["history"].append({"event": "runtime_failure", "tasks": [item["id"] for item in state["tasks"].values() if item["status"] == "runtime_failed"]})
        campaign.save_state(state)
        return "failed"
    stage = state["stage"]
    kind = stage["kind"]
    index = int(stage["interface_index"])
    _, min_successes, max_trials, _ = stage_limits(campaign, kind, index)
    stats = state["statistics"]["flux"] if kind == "flux" else state["statistics"]["interfaces"].setdefault(str(index), {"trials": 0, "successes": 0})
    if stats["successes"] >= min_successes:
        if kind == "flux":
            state["stage"] = {"kind": "interface", "interface_index": 0, "round": 0}
        elif index + 1 >= len(campaign.interfaces) - 1:
            state["phase"] = "complete"
            write_summary(campaign, state)
        else:
            state["stage"] = {"kind": "interface", "interface_index": index + 1, "round": 0}
        state["phase"] = "ready" if state["phase"] != "complete" else "complete"
        state["history"].append({"event": "advanced", "from": stage_name(kind, index), "statistics": stats})
    elif stats["trials"] >= max_trials:
        state["phase"] = "failed"
        state["history"].append({"event": "insufficient_successes", "stage": stage_name(kind, index), "statistics": stats})
    else:
        state["phase"] = "ready"
        state["stage"]["round"] = int(state["stage"].get("round", 0)) + 1
        state["history"].append({"event": "needs_another_batch", "stage": stage_name(kind, index), "statistics": stats})
    campaign.save_state(state)
    return state["phase"]


def write_summary(campaign: Campaign, state: dict[str, Any]) -> None:
    flux = state["statistics"]["flux"]
    timestep = float(cfg_value(campaign.config, "md", "timestep_fs", default=1.0))
    time_ps = flux["steps"] * timestep / 1000.0
    probabilities: list[dict[str, Any]] = []
    product = 1.0
    for index in range(len(campaign.interfaces) - 1):
        stats = state["statistics"]["interfaces"].get(str(index), {"trials": 0, "successes": 0})
        probability = stats["successes"] / stats["trials"] if stats["trials"] else 0.0
        product *= probability
        probabilities.append({"interface_index": index, "from": campaign.interfaces[index], "to": campaign.interfaces[index + 1], **stats, "probability": probability})
    summary = {
        "flux": {**flux, "time_ps_estimate": time_ps, "per_ps": flux["successes"] / time_ps if time_ps else 0.0},
        "conditional_probabilities": probabilities,
        "product_probability": product,
        "rate_per_ps_estimate": (flux["successes"] / time_ps if time_ps else 0.0) * product,
        "note": "The flux time uses configured limits; inspect native logs for exact first-passage durations.",
    }
    write_json(campaign.root / "summary.json", summary)


def simulate_task(campaign: Campaign, task: dict[str, Any], rng: random.Random, outcome: str | None = None) -> None:
    task_dir = Path(task["work_dir"])
    script = sanitize_lsf_for_local_shell((task_dir / "job.lsf").read_text(encoding="utf-8"))
    # Validate the body and LSF metadata rendering without executing a cluster binary.
    validate_simulated_shell(script)
    if outcome is None:
        probability = float(cfg_value(campaign.config, "simulation", "success_probability", default=0.7))
        outcome = "success" if rng.random() < probability else "failure"
    value = float(task["target"]) + 0.01 * campaign.direction if outcome == "success" else float(task["fail"]) - 0.01 * campaign.direction
    initial_cv = 0.0 if task["kind"] == "flux" else campaign.interfaces[int(task["interface_index"])]
    if campaign.backend == "gpu_sampling_ffs":
        status = 1 if outcome == "success" else -1
        (task_dir / task["cv_output"]).write_text(f"0\t0\n{status}\t{value}\n", encoding="utf-8")
        if outcome == "success":
            shutil.copy2(task_dir / "model.xyz", task_dir / "restart.xyz")
    elif campaign.backend == "gpu_sampling_monitor":
        (task_dir / task["cv_output"]).write_text(f"{initial_cv}\n{value}\n", encoding="utf-8")
        shutil.copy2(task_dir / "model.xyz", task_dir / "dump.xyz")
    else:
        (task_dir / task["cv_output"]).write_text(f"#! FIELDS time lambda\n0 {initial_cv}\n1 {value}\n", encoding="utf-8")
        dump = "\n".join([
            "ITEM: TIMESTEP", "1", "ITEM: NUMBER OF ATOMS", "1", "ITEM: BOX BOUNDS pp pp pp",
            "0 10", "0 10", "0 10", "ITEM: ATOMS id type x y z vx vy vz", "1 1 1 1 1 0 0 0", "",
        ])
        (task_dir / task["trajectory"]).write_text(dump, encoding="utf-8")
    (task_dir / "ffs_runtime.out").write_text("1 steps completed.\n", encoding="utf-8")
    (task_dir / ".ffs_exit_code").write_text("0\n", encoding="utf-8")


def simulate(campaign: Campaign, seed: int, maximum_cycles: int = 100) -> str:
    rng = random.Random(seed)
    pattern = list(cfg_value(campaign.config, "simulation", "outcomes", default=[]))
    pattern_index = 0
    for _ in range(maximum_cycles):
        state = campaign.load_state()
        if state["phase"] in {"complete", "failed"}:
            return state["phase"]
        if not [task for task in state["tasks"].values() if task["status"] in {"planned", "submitted"}]:
            plan(campaign)
            state = campaign.load_state()
        for task in state["tasks"].values():
            if task["status"] == "planned":
                forced = pattern[pattern_index % len(pattern)] if pattern else None
                pattern_index += 1
                simulate_task(campaign, task, rng, forced)
        collect(campaign)
        advance(campaign)
    raise RuntimeError("Simulation exceeded maximum cycles")


def run_local(campaign: Campaign, *, seed: int, random_crossings: bool,
              success_probability: float) -> list[dict[str, Any]]:
    """Run rendered jobs locally for integration testing, never through LSF.

    This is intended to be invoked from the Linux environment that owns the
    executable.  It executes the generated job body, so .ffs_exit_code and all
    physics outputs come from the real process.  Random crossing decisions are
    stored separately and only affect the subsequent collector classification.
    """
    state = campaign.load_state()
    if random_crossings and campaign.backend == "gpu_sampling_ffs":
        raise ValueError("run-local --random-crossings requires gpu_sampling_monitor or lammps_plumed")
    results: list[dict[str, Any]] = []
    for task in state["tasks"].values():
        if task["status"] != "planned":
            continue
        task_dir = Path(task["work_dir"])
        script = sanitize_lsf_for_local_shell((task_dir / "job.lsf").read_text(encoding="utf-8"))
        completed = subprocess.run(["bash", "-c", script], cwd=task_dir, text=True,
                                   capture_output=True, check=False)
        if completed.returncode and not (task_dir / ".ffs_exit_code").exists():
            (task_dir / ".ffs_exit_code").write_text(f"{completed.returncode}\n", encoding="utf-8")
        result = {"task": task["id"], "return_code": completed.returncode}
        if random_crossings and completed.returncode == 0:
            result["decision"] = local_crossing_decision(campaign, task, seed, success_probability)
        results.append(result)
    return results


def status(campaign: Campaign) -> dict[str, Any]:
    state = campaign.load_state()
    buckets: dict[str, int] = {}
    for task in state["tasks"].values():
        buckets[task["status"]] = buckets.get(task["status"], 0) + 1
    return {"phase": state["phase"], "stage": state["stage"], "tasks": buckets, "root": str(campaign.root)}


def cycle(campaign: Campaign, submit_jobs: bool) -> dict[str, Any]:
    collected = collect(campaign)
    current = campaign.load_state()
    if collected["waiting"]:
        return {"action": "waiting", **collected, "status": status(campaign)}
    if current["phase"] == "running":
        advance(campaign)
    current = campaign.load_state()
    if current["phase"] in {"complete", "failed"}:
        return {"action": current["phase"], "status": status(campaign)}
    tasks = plan(campaign)
    submitted = submit(campaign) if submit_jobs else []
    return {"action": "planned", "tasks": [task["id"] for task in tasks], "submitted": submitted, "status": status(campaign)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage-wise LSF FFS manager for GPU-Sampling and LAMMPS+PLUMED")
    parser.add_argument("config", type=Path, help="remote-runner YAML configuration")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="create an empty campaign state")
    sub.add_parser("plan", help="generate one current-stage batch")
    sub.add_parser("submit", help="submit all planned jobs through bsub")
    sub.add_parser("collect", help="collect terminal task outputs")
    sub.add_parser("advance", help="advance after a fully collected stage")
    cycle_parser = sub.add_parser("cycle", help="collect, advance, plan, and optionally submit one stage")
    cycle_parser.add_argument("--submit", action="store_true", help="call bsub after planning")
    sim_parser = sub.add_parser("simulate", help="run deterministic local scheduler/output simulation")
    sim_parser.add_argument("--seed", type=int, default=20260808)
    sim_parser.add_argument("--max-cycles", type=int, default=100)
    local_parser = sub.add_parser("run-local", help="execute planned jobs locally for integration testing")
    local_parser.add_argument("--seed", type=int, default=20260808, help="seed for test-only crossing decisions")
    local_parser.add_argument("--random-crossings", action="store_true",
                              help="classify completed monitor trajectories with a deterministic random rule")
    local_parser.add_argument("--success-probability", type=float, default=0.70,
                              help="test-only success probability when --random-crossings is used")
    sub.add_parser("status", help="show campaign and task counts")
    args = parser.parse_args(argv)
    campaign = Campaign(args.config)
    if args.command == "init":
        result = campaign.init()
    elif args.command == "plan":
        result = {"tasks": [task["id"] for task in plan(campaign)]}
    elif args.command == "submit":
        result = {"submitted": submit(campaign)}
    elif args.command == "collect":
        result = collect(campaign)
    elif args.command == "advance":
        result = {"phase": advance(campaign)}
    elif args.command == "cycle":
        result = cycle(campaign, args.submit)
    elif args.command == "simulate":
        result = {"phase": simulate(campaign, args.seed, args.max_cycles), "status": status(campaign)}
    elif args.command == "run-local":
        result = {"ran": run_local(campaign, seed=args.seed, random_crossings=args.random_crossings,
                                    success_probability=args.success_probability)}
    else:
        result = status(campaign)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
