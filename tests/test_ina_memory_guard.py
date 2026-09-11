#!/usr/bin/env python3
"""Short GPU regressions; artificial reserve limits exercise rejection without OOM."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess


def frames(path):
    count = 0
    with path.open() as handle:
        while True:
            line = handle.readline()
            if not line:
                return count
            n = int(line)
            for _ in range(n + 1):
                assert handle.readline(), "truncated XYZ"
            count += 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpumd", type=Path, required=True)
    p.add_argument("--initial", type=Path, required=True)
    p.add_argument("--final", type=Path, required=True)
    p.add_argument("--potential", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    results = []
    tests = [
        ("allowed", "ina_memory_guard 1 ina_memory_reserve_mb 0", 1, "", True),
        ("disabled", "ina_memory_guard 0 ina_memory_reserve_mb 1000000000", 1, "", True),
        ("blocked", "ina_memory_guard 1 ina_memory_reserve_mb 1000000000", 1, "", True),
        ("fixed_blocked", "ina_memory_reserve_mb 1000000000", 1, "neb_set no_vc match_cell_to_initial\n", True),
        ("bootstrap_blocked", "ina_memory_reserve_mb 1000000000", 0, "", False),
        ("bad_reserve", "ina_memory_reserve_mb -1", 1, "", False),
        ("bad_switch", "ina_memory_guard 2", 1, "", False),
        ("remove_only", "ina_memory_reserve_mb 1000000000", 3, "", True),
    ]
    for name, options, interpolate, extra, success in tests:
        work = a.output / name
        work.mkdir()
        shutil.copy2(a.initial, work / "is.xyz")
        shutil.copy2(a.final, work / "fs.xyz")
        (work / "nep.txt").symlink_to(a.potential.resolve())
        distance = "10 20" if name == "remove_only" else "0.000001 0.000002"
        (work / "run.in").write_text(
            "potential nep.txt\n"
            f"neb_set is_name is.xyz fs_name fs.xyz interpolate {interpolate}\n"
            f"neb_set {options}\n" + extra +
            "neb_set tangent modified\n"
            f"neb_set dist_range {distance} dist_ncount 5 ina_interval 1\n"
            "neb_set ina_force_tol 1 1e100\n"
            "neb_set ina_local_relax_steps 2 ina_local_relax_neighbors 1\n"
            "neb_set print_interval 1 dump_interval 10000 peek_interval 10000\n"
            "neb_set k 0.1 p 0\n"
            "neb_run fire 1e-12 8 dt_max 1 dt_min 0.005 max_move 0.05 imagewise\n")
        with (work / "gpumd.log").open("w") as log:
            r = subprocess.run([str(a.gpumd.resolve())], cwd=work, stdout=log, stderr=subprocess.STDOUT)
        text = (work / "gpumd.log").read_text()
        assert (r.returncode == 0) == success, (name, r.returncode)
        if not success:
            expected = {"bootstrap_blocked": "cannot allocate the first intermediate image",
                        "bad_reserve": "finite non-negative", "bad_switch": "should be 0 or 1"}[name]
            assert expected in text, name
            assert not (work / "final_traj.xyz").exists(), name
            results.append(dict(case=name, expected_rejection=True))
            continue
        n = frames(work / "final_traj.xyz")
        summary = re.search(r"NEB step summary: regular=(\d+), INA local=(\d+), total=(\d+)", text)
        assert summary and int(summary[1]) == 8, name
        if name in ("blocked", "fixed_blocked"):
            assert n == 3 and "insertion skipped" in text, name
            assert int(summary[2]) == 0, name
            assert text.count("New minimization,") == 1, name
        elif name in ("allowed", "disabled"):
            assert n > 3 and "add an image:" in text, name
            assert "insertion skipped" not in text, name
            assert int(summary[2]) > 0, name
        else:
            assert n < 5 and "remove an image:" in text, name
        results.append(dict(case=name, images=n, regular=int(summary[1]), local=int(summary[2])))
    (a.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
