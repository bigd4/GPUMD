"""Build the CVOnlyProcedure model used by the GPU-Sampling Monitor test."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


PYGAS_ROOT = Path(__file__).resolve().parents[4] / "pygas"
if str(PYGAS_ROOT) not in sys.path:
    sys.path.insert(0, str(PYGAS_ROOT))

from base.combine import CVOnlyProcedure


class FirstAtomX(CVOnlyProcedure):
    """One scalar CV with the production CVOnlyProcedure dictionary ABI."""

    def __init__(self) -> None:
        super().__init__({})
        self.cv_dim = 1

    def _cal_cv(self, positions, cell, side_array):
        return positions[0, 0].reshape(1)


if __name__ == "__main__":
    output = Path(__file__).with_name("cvonly-first-atom-x.pt")
    model = FirstAtomX().to(dtype=torch.double).eval()
    torch.jit.script(model).save(str(output))
    print(f"Saved {output}")
