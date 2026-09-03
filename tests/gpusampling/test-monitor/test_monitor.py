import math
import os
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
BASE_CASE_DIR = REPO_ROOT / "tests" / "gpusampling" / "test-metad"
CASE_DIR = Path(__file__).resolve().parent
RUN_STEPS = 20


def _gpu_sampling_executable() -> Path:
    executable = os.environ.get("GPU_SAMPLING_EXECUTABLE") or os.environ.get("GPUMD_EXECUTABLE")
    if executable:
        return Path(executable).resolve()

    executable_name = "gpu-sampling.exe" if os.name == "nt" else "gpu-sampling"
    for candidate in (
        REPO_ROOT / "src" / executable_name,
        REPO_ROOT / "src" / "build" / executable_name,
    ):
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Set GPU_SAMPLING_EXECUTABLE to the GPU-Sampling executable before running this test."
    )


def _run_process(case_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(_gpu_sampling_executable())],
        cwd=case_dir,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


def _run(case_dir: Path) -> str:
    completed = _run_process(case_dir)
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    return output


def _prepare_case(tmp_path: Path, name: str) -> Path:
    case_dir = tmp_path / name
    shutil.copytree(BASE_CASE_DIR, case_dir)
    shutil.copy(CASE_DIR / "cvonly-first-atom-x.pt", case_dir)
    shutil.copy(CASE_DIR / "alanine-torsion-cvonly.pt", case_dir)
    shutil.copy(CASE_DIR / "monitor-default.yaml", case_dir)
    shutil.copy(CASE_DIR / "monitor-custom.yaml", case_dir)
    shutil.copy(CASE_DIR / "monitor-matching.yaml", case_dir)
    shutil.copy(CASE_DIR / "metad-matching.yaml", case_dir)
    shutil.copy(CASE_DIR / "run-metad-monitor.in", case_dir)
    return case_dir


def _read_rows(path: Path, columns: int) -> list[list[float]]:
    rows = []
    for line in path.read_text().splitlines():
        values = [float(value) for value in line.split()]
        assert len(values) == columns, f"Unexpected row in {path}: {line!r}"
        assert all(math.isfinite(value) for value in values)
        rows.append(values)
    return rows


def _write_plain_md_run(case_dir: Path, monitor_line: str = "") -> None:
    lines = [
        # A fixed seed makes the control and monitored trajectories directly
        # comparable; the Monitor must not perturb MD state.
        "velocity 300 seed 12345",
        "potential nep-alanine.txt",
        "time_step 1",
        "ensemble nve",
        "dump_thermo 1",
    ]
    if monitor_line:
        lines.append(monitor_line)
    lines.append(f"run {RUN_STEPS}")
    (case_dir / "run.in").write_text("\n".join(lines) + "\n")


def test_monitor_default_log_is_pure_cv(tmp_path):
    case_dir = _prepare_case(tmp_path, "monitor-default")
    _write_plain_md_run(case_dir, "Monitor cvonly-first-atom-x.pt monitor-default.yaml")

    output = _run(case_dir)

    assert "CVonly model loaded successfully from cvonly-first-atom-x.pt" in output
    rows = _read_rows(case_dir / "MonCVlog.txt", columns=1)
    assert len(rows) == RUN_STEPS // 5 + 1
    assert not (case_dir / "GASCVlog.txt").exists()


def test_metad_and_monitor_keep_separate_custom_logs(tmp_path):
    case_dir = _prepare_case(tmp_path, "separate-logs")
    metad_config = (BASE_CASE_DIR / "GAScfg.yaml").read_text() + "cv_log_path: metad-cvlog.txt\n"
    (case_dir / "metad-custom.yaml").write_text(metad_config)
    (case_dir / "run.in").write_text(
        "\n".join(
            [
                "velocity 300",
                "potential nep-alanine.txt",
                "time_step 1",
                "MetaD Alanine-torsion-pd-02.pt metad-custom.yaml",
                "Monitor cvonly-first-atom-x.pt monitor-custom.yaml",
                "ensemble nvt_bdp 300 300 100",
                f"run {RUN_STEPS}",
            ]
        )
        + "\n"
    )

    _run(case_dir)

    assert len(_read_rows(case_dir / "metad-cvlog.txt", columns=4)) == 3
    assert len(_read_rows(case_dir / "monitor-cvlog.txt", columns=1)) == 3
    assert not (case_dir / "GASCVlog.txt").exists()
    assert not (case_dir / "MonCVlog.txt").exists()


def test_monitor_matches_the_metad_cv_definition(tmp_path):
    case_dir = _prepare_case(tmp_path, "matching-torsions")
    shutil.copy(case_dir / "run-metad-monitor.in", case_dir / "run.in")

    _run(case_dir)

    metad_rows = _read_rows(case_dir / "metad-torsion-cvlog.txt", columns=4)
    monitor_rows = _read_rows(case_dir / "monitor-torsion-cvlog.txt", columns=2)
    assert len(metad_rows) == len(monitor_rows) == 3
    for metad, monitor in zip(metad_rows, monitor_rows):
        assert math.isclose(metad[1], monitor[0], rel_tol=0.0, abs_tol=1.0e-12)
        assert math.isclose(metad[2], monitor[1], rel_tol=0.0, abs_tol=1.0e-12)


def test_monitor_does_not_change_md_observables(tmp_path):
    control_dir = _prepare_case(tmp_path, "control")
    monitored_dir = _prepare_case(tmp_path, "monitored")
    _write_plain_md_run(control_dir)
    _write_plain_md_run(monitored_dir, "Monitor cvonly-first-atom-x.pt monitor-default.yaml")

    _run(control_dir)
    _run(monitored_dir)

    assert (control_dir / "thermo.out").read_text() == (monitored_dir / "thermo.out").read_text()
    assert len(_read_rows(monitored_dir / "MonCVlog.txt", columns=1)) == RUN_STEPS // 5 + 1


def test_monitor_reports_neighbor_capacity_overflow(tmp_path):
    case_dir = _prepare_case(tmp_path, "neighbor-overflow")
    config = (CASE_DIR / "monitor-default.yaml").read_text().replace(
        "max_neighbors: 64", "max_neighbors: 1"
    )
    (case_dir / "monitor-overflow.yaml").write_text(config)
    _write_plain_md_run(case_dir, "Monitor cvonly-first-atom-x.pt monitor-overflow.yaml")

    completed = _run_process(case_dir)
    output = completed.stdout + completed.stderr

    assert completed.returncode != 0
    assert "Monitor neighbor list exceeded max_neighbors" in output


def test_small_periodic_cell_emits_each_directed_neighbor_once(tmp_path):
    case_dir = _prepare_case(tmp_path, "small-periodic-cell")
    n_atoms = int((case_dir / "model.xyz").read_text().splitlines()[0])
    config = (CASE_DIR / "monitor-default.yaml").read_text()
    config = config.replace("neighbor_rc: 5.5", "neighbor_rc: 100.0")
    config = config.replace("max_neighbors: 64", f"max_neighbors: {n_atoms - 1}")
    (case_dir / "monitor-small-cell.yaml").write_text(config)
    _write_plain_md_run(case_dir, "Monitor cvonly-first-atom-x.pt monitor-small-cell.yaml")

    _run(case_dir)

    assert len(_read_rows(case_dir / "MonCVlog.txt", columns=1)) == RUN_STEPS // 5 + 1
