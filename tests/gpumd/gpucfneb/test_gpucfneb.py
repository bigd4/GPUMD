import math
import os
import re
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_DIR = REPO_ROOT / "examples" / "gpucfneb" / "diamond_vacancy"
NEB_STEPS = 5


def _gpumd_executable() -> Path:
    executable = os.environ.get("GPUMD_EXECUTABLE")
    return Path(executable).resolve() if executable else REPO_ROOT / "src" / "gpumd"


def _read_xyz_frame_sizes(filename: Path) -> list[int]:
    lines = filename.read_text().splitlines()
    frame_sizes = []
    offset = 0
    while offset < len(lines):
        number_of_atoms = int(lines[offset])
        assert offset + number_of_atoms + 2 <= len(lines), f"Incomplete XYZ frame in {filename}"
        for atom_line in lines[offset + 2 : offset + number_of_atoms + 2]:
            fields = atom_line.split()
            assert len(fields) >= 4
            assert all(math.isfinite(float(value)) for value in fields[1:4])
        frame_sizes.append(number_of_atoms)
        offset += number_of_atoms + 2
    assert offset == len(lines), f"Incomplete XYZ frame in {filename}"
    return frame_sizes


def test_diamond_vacancy_example_runs(tmp_path):
    case_dir = tmp_path / "diamond_vacancy"
    shutil.copytree(EXAMPLE_DIR, case_dir)
    run_input = case_dir / "run.in"
    run_text = run_input.read_text()
    original_command = "neb_run fire 0.01 1000"
    assert original_command in run_text
    run_input.write_text(run_text.replace(original_command, f"neb_run fire 0.01 {NEB_STEPS}", 1))

    completed = subprocess.run(
        [str(_gpumd_executable())],
        cwd=case_dir,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    assert "ENTERING NEB CALCULATION" in output

    summary = re.search(
        r"NEB step summary: regular=(\d+), INA local=(\d+), total=(\d+)\.",
        output,
    )
    assert summary is not None, output
    regular_steps, local_steps, total_steps = map(int, summary.groups())
    assert regular_steps == NEB_STEPS
    assert local_steps == 0
    assert total_steps == regular_steps

    step_records = re.findall(r"step:\s*(\d+),.*fmax=\s*([0-9.eE+-]+)", output)
    assert len(step_records) == NEB_STEPS
    assert all(math.isfinite(float(fmax)) for _, fmax in step_records)

    energy_file = case_dir / "neb_energies.out"
    energies = [float(line) for line in energy_file.read_text().splitlines()]
    assert len(energies) == 7
    assert all(math.isfinite(energy) for energy in energies)
    assert _read_xyz_frame_sizes(case_dir / "final_traj.xyz") == [63] * 7
