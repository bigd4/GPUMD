#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: smoke_aoti_routes.sh GAS_AOTI_SMOKE SPEC_FILE" >&2
  echo "spec columns: model route n_atoms cv_size max_edges max_history [opes_capacity]" >&2
  exit 2
fi
runner=$1
spec=$2
[[ -x "$runner" ]] || { echo "Missing smoke executable: $runner" >&2; exit 1; }

while read -r model route n_atoms cv_size max_edges max_history opes_capacity extra; do
  [[ -z "${model:-}" || "$model" == \#* ]] && continue
  [[ -z "${extra:-}" ]] || { echo "Too many columns for $model" >&2; exit 1; }
  command=("$runner" "$model" "$route" "$n_atoms" "$cv_size" "$max_edges" "$max_history")
  [[ -z "${opes_capacity:-}" ]] || command+=("$opes_capacity")
  "${command[@]}"
done < "$spec"
