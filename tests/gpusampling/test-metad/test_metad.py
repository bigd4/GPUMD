import math
import os
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
CASE_DIR = Path(__file__).resolve().parent
RUN_STEPS = 100
CV_LOG_INTERVAL = 10
ANGLE_TOLERANCE = 0.5
SCALAR_TOLERANCE = 1.0e-12


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


def _read_cv_log(filename: Path) -> list[list[float]]:
    rows = []
    for line in filename.read_text().splitlines():
        values = [float(value) for value in line.split()]
        assert len(values) == 4, f"Unexpected CV log row in {filename}: {line!r}"
        assert all(math.isfinite(value) for value in values)
        rows.append(values)
    return rows


def _periodic_interval(values: list[float]) -> tuple[float, float]:
    """Return the shortest 2π-periodic interval that contains all values."""
    period = 2.0 * math.pi
    wrapped = sorted((value + math.pi) % period - math.pi for value in values)
    gaps = [wrapped[index + 1] - wrapped[index] for index in range(len(wrapped) - 1)]
    gaps.append(wrapped[0] + period - wrapped[-1])
    largest_gap_index = max(range(len(gaps)), key=gaps.__getitem__)
    start = wrapped[(largest_gap_index + 1) % len(wrapped)]
    end = wrapped[largest_gap_index] + period
    return start, end


def _unwrap_near(value: float, reference_center: float) -> float:
    period = 2.0 * math.pi
    return value + round((reference_center - value) / period) * period


def _assert_periodic_range_matches(
    actual_rows: list[list[float]], reference_rows: list[list[float]], column: int
) -> None:
    reference_start, reference_end = _periodic_interval([row[column] for row in reference_rows])
    reference_center = 0.5 * (reference_start + reference_end)
    actual_values = [_unwrap_near(row[column], reference_center) for row in actual_rows]
    actual_start = min(actual_values)
    actual_end = max(actual_values)
    assert actual_start >= reference_start - ANGLE_TOLERANCE and actual_end <= reference_end + ANGLE_TOLERANCE, (
        f"CV {column} range drifted: actual=[{actual_start:.6f}, {actual_end:.6f}], "
        f"reference=[{reference_start:.6f}, {reference_end:.6f}], "
        f"tolerance={ANGLE_TOLERANCE:.6f} rad"
    )


def _assert_scalar_range_matches(
    actual_rows: list[list[float]], reference_rows: list[list[float]], column: int
) -> None:
    actual_range = (min(row[column] for row in actual_rows), max(row[column] for row in actual_rows))
    reference_range = (min(row[column] for row in reference_rows), max(row[column] for row in reference_rows))
    assert (
        actual_range[0] >= reference_range[0] - SCALAR_TOLERANCE
        and actual_range[1] <= reference_range[1] + SCALAR_TOLERANCE
    ), (
        f"Scalar {column} range drifted: actual={actual_range}, reference={reference_range}, "
        f"tolerance={SCALAR_TOLERANCE}"
    )


def test_metad_torchscript_model_runs(tmp_path):
    case_dir = tmp_path / "test-metad"
    shutil.copytree(CASE_DIR, case_dir)

    completed = subprocess.run(
        [str(_gpu_sampling_executable())],
        cwd=case_dir,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    output = completed.stdout + completed.stderr

    assert completed.returncode == 0, output
    assert "GASCVModel loaded successfully from Alanine-torsion-pd-02.pt" in output
    assert f"Run {RUN_STEPS} steps." in output
    assert "Finished running GPU-Sampling." in output

    actual_rows = _read_cv_log(case_dir / "GASCVlog.txt")
    reference_rows = _read_cv_log(case_dir / "GASCVlog1.txt")
    assert len(actual_rows) == RUN_STEPS // CV_LOG_INTERVAL + 1
    _assert_scalar_range_matches(actual_rows, reference_rows, 0)
    _assert_periodic_range_matches(actual_rows, reference_rows, 1)
    _assert_periodic_range_matches(actual_rows, reference_rows, 2)
    _assert_scalar_range_matches(actual_rows, reference_rows, 3)
