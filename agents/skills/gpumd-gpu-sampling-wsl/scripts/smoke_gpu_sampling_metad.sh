#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 GPU_SAMPLING_EXECUTABLE GPUMD_METAD_TEST_CASE" >&2
  exit 2
fi

executable="$1"
case_dir="$2"
if [[ ! -x "$executable" ]]; then
  echo "GPU-Sampling executable is not executable: $executable" >&2
  exit 2
fi
if [[ ! -f "$case_dir/run.in" || ! -f "$case_dir/Alanine-torsion-pd-02.pt" ]]; then
  echo "Not a complete GPU-Sampling MetaD test case: $case_dir" >&2
  exit 2
fi

work_dir="$(mktemp -d "${TMPDIR:-/tmp}/gpu-sampling-metad.XXXXXX")"
cleanup() { rm -rf "$work_dir"; }
trap cleanup EXIT
cp -a "$case_dir"/. "$work_dir"/

set +e
(cd "$work_dir" && "$executable") >"$work_dir/gpu-sampling.log" 2>&1
status=$?
set -e
cat "$work_dir/gpu-sampling.log"
if [[ $status -ne 0 ]]; then
  echo "GPU-Sampling exited with status $status" >&2
  exit "$status"
fi
grep -Fq 'GASCVModel loaded successfully' "$work_dir/gpu-sampling.log"
grep -Fq 'Finished running GPU-Sampling.' "$work_dir/gpu-sampling.log"
[[ -s "$work_dir/GASCVlog.txt" ]]
echo "GPU-Sampling MetaD smoke test passed."
