"""Local, no-LSF integration coverage for the remote FFS manager."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


RUNNER = Path(__file__).resolve().parents[1] / "ffs_remote_runner.py"
USER_GPU_SAMPLE = Path(r"D:\data\calculations\metad\cparad\template")


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def run(config: Path, *arguments: str) -> dict:
    completed = subprocess.run([sys.executable, str(RUNNER), str(config), *arguments], text=True, capture_output=True)
    if completed.returncode:
        raise AssertionError(f"runner failed\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}")
    return json.loads(completed.stdout)


def make_common_files(root: Path) -> None:
    write(root / "job.template", "#!/bin/bash\n! #BSUB -q disabled\n#BSUB -q oldqueue\n! #BSUB -q disabled-alt\n#BSUB -gpu \"num=1:aff=yes\"\n#BSUB -J old-name\n#BSUB -o out\n#BSUB -e err\nmodule load cuda/test\n{{FFS_COMMAND}}\n")
    write(root / "model.xyz", "1\nProperties=species:S:1:pos:R:3:vel:R:3 Lattice=\"10 0 0 0 10 0 0 0 10\" pbc=\"T T T\"\nC 1 1 1 0 0 0\n")
    write(root / "nep.txt", "fake potential\n")
    write(root / "cv.pt", "fake torchscript fixture\n")
    write(root / "GAScfg.yaml", "cv_log_interval: 1\nneighbor_rc: 3.0\nmax_neighbors: 8\n")
    executable = write(root / "fake-executable", "#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)


class RemoteRunnerTest(unittest.TestCase):
    @unittest.skipUnless(USER_GPU_SAMPLE.is_dir(), "user GPU-Sampling LSF template is not mounted")
    def test_user_cparad_lsf_template_renders_without_running_md(self) -> None:
        """Exercise the supplied LSF/run.in/GAScfg assets in local simulation mode."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = write(root / "fake-executable", "#!/bin/sh\nexit 0\n")
            executable.chmod(0o755)
            config = write(root / "campaign.yaml", f"campaign:\n  name: cparad\n  root: campaign\nscheduler:\n  job_template: {USER_GPU_SAMPLE / 'job'}\n  queue: sample-queue\nexecution:\n  backend: gpu_sampling_ffs\n  mode: gpu\n  executable: fake-executable\n  run_template: {USER_GPU_SAMPLE / 'run.in'}\nresources:\n  initial_states: [{USER_GPU_SAMPLE / 'model-seed.xyz'}]\n  nep: {USER_GPU_SAMPLE / 'nep.txt'}\n  pt: {USER_GPU_SAMPLE / 'carbon-parad-diamond-ann-cv.pt'}\n  gas_config: {USER_GPU_SAMPLE / 'GAScfg.yaml'}\nffs:\n  interfaces: [0.1, 0.2]\n  fail_interface: 0.0\n  randomize_velocities: true\n  flux: {{batch_size: 1, min_successes: 1, max_trials: 1, steps: 2}}\n  interface: {{batch_size: 1, min_successes: 1, max_trials: 1, steps: 2}}\nmd:\n  temperature_K: 300\n")
            run(config, "init")
            planned = run(config, "plan")
            self.assertEqual(len(planned["tasks"]), 1)
            task_dir = next((root / "campaign" / "tasks" / "flux").glob("attempt_*"))
            job = (task_dir / "job.lsf").read_text(encoding="utf-8")
            self.assertIn("! #BSUB -q 734090ib", job)
            self.assertIn("#BSUB -q sample-queue", job)
            self.assertIn("#BSUB -J cparad-0-1", job)
            self.assertIn("-m model.xyz -i run.in", job)
            generated = (task_dir / "run.in").read_text(encoding="utf-8")
            self.assertIn("potential nep.txt", generated)
            self.assertIn("FFSampling carbon-parad-diamond-ann-cv.pt GAScfg.ffs.yaml", generated)
            self.assertIn("n_atoms: 36000", (task_dir / "GAScfg.ffs.yaml").read_text(encoding="utf-8"))

    def test_gpu_ffs_without_velocity_randomization_discards_failed_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_common_files(root)
            write(root / "run.in", "potential old-nep.txt\nGASMD wrong.pt wrong.yaml\ntime_step 1\nvelocity 300 seed 5\nensemble nve\nrun 10\n")
            config = write(root / "campaign.yaml", "campaign:\n  name: serial-no-resample\n  root: campaign\nscheduler:\n  job_template: job.template\n  queue: actualqueue\nexecution:\n  backend: gpu_sampling_ffs\n  mode: gpu\n  executable: fake-executable\n  run_template: run.in\nresources:\n  initial_states: [model.xyz]\n  nep: nep.txt\n  pt: cv.pt\n  gas_config: GAScfg.yaml\nffs:\n  interfaces: [0.1, 0.2, 0.3]\n  direction: 1\n  fail_interface: 0.0\n  randomize_velocities: false\n  replicas_per_crossing: 9\n  flux:\n    batch_size: 1\n    min_successes: 3\n    max_trials: 3\n    steps: 2\n  interface:\n    batch_size: 3\n    min_successes: 2\n    max_trials: 3\n    steps: 2\nsimulation:\n  outcomes: [success, success, success, success, failure, success, success, success]\n")
            run(config, "init")
            result = run(config, "simulate", "--seed", "1")
            self.assertEqual(result["phase"], "complete")
            self.assertTrue((root / "campaign" / "summary.json").is_file())
            state = json.loads((root / "campaign" / "campaign_state.json").read_text(encoding="utf-8"))
            first_interface_tasks = [task for task in state["tasks"].values() if task["kind"] == "interface" and task["interface_index"] == 0]
            self.assertEqual(len(first_interface_tasks), 3)
            self.assertEqual(len({task["start_state_id"] for task in first_interface_tasks}), 3)
            self.assertEqual(sum(task["status"] == "failure" for task in first_interface_tasks), 1)
            consumed = {record["id"]: record.get("consumed_by") for record in state["pools"]["0"]}
            for task in first_interface_tasks:
                self.assertEqual(consumed[task["start_state_id"]], task["id"])
            job = Path(first_interface_tasks[0]["work_dir"]) / "job.lsf"
            rendered = job.read_text(encoding="utf-8")
            self.assertIn("! #BSUB -q disabled", rendered)
            self.assertIn("#BSUB -q actualqueue", rendered)
            self.assertIn("#BSUB -J serial-no-resample-1-", rendered)
            self.assertIn("-m model.xyz -i run.in", rendered)
            run_input = (job.parent / "run.in").read_text(encoding="utf-8")
            self.assertIn("potential nep.txt", run_input)
            self.assertIn("FFSampling cv.pt GAScfg.ffs.yaml", run_input)
            self.assertNotIn("GASMD wrong.pt", run_input)

    def test_gpu_cvonly_monitor_uses_configured_output_and_crossing_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_common_files(root)
            write(root / "run.in", "potential stale.txt\nvelocity 300\nensemble nve\nrun 4\n")
            config = write(root / "campaign.yaml", "campaign:\n  name: monitor\n  root: campaign\nscheduler:\n  job_template: job.template\n  queue: gpuq\nexecution:\n  backend: gpu_sampling_monitor\n  mode: gpu\n  executable: fake-executable\n  run_template: run.in\nresources:\n  initial_states: [model.xyz]\n  nep: nep.txt\n  pt: cv.pt\n  gas_config: GAScfg.yaml\nmonitor:\n  cv_index: 0\n  output: selected-cv.log\n  stride: 1\nffs:\n  interfaces: [0.1, 0.2]\n  fail_interface: 0.0\n  randomize_velocities: true\n  replicas_per_crossing: 1\n  flux: {batch_size: 1, min_successes: 1, max_trials: 1, steps: 2}\n  interface: {batch_size: 1, min_successes: 1, max_trials: 1, steps: 2}\nsimulation:\n  outcomes: [success]\n")
            run(config, "init")
            self.assertEqual(run(config, "simulate")["phase"], "complete")
            state = json.loads((root / "campaign" / "campaign_state.json").read_text(encoding="utf-8"))
            first_task = next(iter(state["tasks"].values()))
            task_dir = Path(first_task["work_dir"])
            self.assertTrue((task_dir / "selected-cv.log").exists())
            self.assertIn("Monitor cv.pt CVMonitor.yaml", (task_dir / "run.in").read_text(encoding="utf-8"))
            self.assertIn("cv_log_path: selected-cv.log", (task_dir / "CVMonitor.yaml").read_text(encoding="utf-8"))

    def test_lammps_plumed_cpu_flow_and_snapshot_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_common_files(root)
            write(root / "model-A.data", "LAMMPS data placeholder\n")
            write(root / "in.lammps", "{{FFS_START_COMMAND}}\npair_style nep\npair_coeff * * {{FFS_NEP_FILE}}\nfix ffs all plumed plumedfile {{FFS_PLUMED_FILE}}\n{{FFS_TRAJECTORY_COMMAND}}\nrun {{FFS_STEPS}}\n")
            write(root / "plumed.dat", "cv: DISTANCE ATOMS=1,2\nPRINT ARG=cv FILE={{FFS_CV_LOG}} STRIDE={{FFS_MONITOR_STRIDE}}\n")
            config = write(root / "campaign.yaml", "campaign:\n  name: lammps\n  root: campaign\nscheduler:\n  job_template: job.template\n  queue: cpuq\n  cpu_cores: 6\nexecution:\n  backend: lammps_plumed\n  mode: cpu\n  executable: fake-executable\nresources:\n  initial_states: [model-A.data]\n  nep: nep.txt\nffs:\n  interfaces: [0.1, 0.2]\n  fail_interface: 0.0\n  randomize_velocities: false\n  flux: {batch_size: 1, min_successes: 1, max_trials: 1, steps: 2}\n  interface: {batch_size: 1, min_successes: 1, max_trials: 1, steps: 2}\nlammps:\n  input_template: in.lammps\n  plumed_template: plumed.dat\n  base_data: model-A.data\n  cv_log: COLVAR\n  cv_column: lambda\n  monitor_stride: 1\n  trajectory: ffs-trajectory.lammpstrj\nsimulation:\n  outcomes: [success]\n")
            run(config, "init")
            self.assertEqual(run(config, "simulate")["phase"], "complete")
            state = json.loads((root / "campaign" / "campaign_state.json").read_text(encoding="utf-8"))
            first_task = next(iter(state["tasks"].values()))
            task_dir = Path(first_task["work_dir"])
            job = (task_dir / "job.lsf").read_text(encoding="utf-8")
            self.assertIn("#BSUB -n 6", job)
            self.assertNotIn("#BSUB -gpu", job)
            rendered = (task_dir / "in.lammps").read_text(encoding="utf-8")
            self.assertIn("read_data model-A.data", rendered)
            self.assertIn("plumedfile plumed.dat", rendered)
            self.assertIn("dump ffs all custom 1 ffs-trajectory.lammpstrj", rendered)
            self.assertTrue((root / "campaign" / "states" / "interface_000").exists())


if __name__ == "__main__":
    unittest.main()
