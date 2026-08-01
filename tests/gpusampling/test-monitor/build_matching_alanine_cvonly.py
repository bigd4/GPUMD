"""Export the CVOnlyProcedure-equivalent of the Alanine MetaD test model.

The source MetaD artifact defines CVs as phi(4, 6, 8, 14) and
psi(6, 8, 14, 16), with zero-based atom indices and output order [phi, psi].
"""

from __future__ import annotations

from typing import Dict
from pathlib import Path

import torch


class MatchingAlanineCVOnly(torch.nn.Module):
    """The CVOnlyProcedure inference ABI with the test MetaD torsions."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("phi_indices", torch.tensor([4, 6, 8, 14], dtype=torch.long))
        self.register_buffer("psi_indices", torch.tensor([6, 8, 14, 16], dtype=torch.long))
        self.cv_dim = 2

    def _torsion(self, coordinates: torch.Tensor, lattice: torch.Tensor, atom_indices: torch.Tensor):
        fractional = coordinates @ torch.linalg.inv(lattice).T
        selected = fractional[atom_indices]
        fractional_vectors = selected[1:] - selected[:-1]
        vectors = (fractional_vectors - torch.round(fractional_vectors)) @ lattice.T
        normal_1 = torch.cross(vectors[0], vectors[1])
        normal_2 = torch.cross(vectors[1], vectors[2])
        normal_1 = normal_1 / (torch.linalg.norm(normal_1) + 1.0e-12)
        normal_2 = normal_2 / (torch.linalg.norm(normal_2) + 1.0e-12)
        bond_2 = vectors[1] / (torch.linalg.norm(vectors[1]) + 1.0e-12)
        return torch.atan2(torch.sum(torch.cross(normal_1, bond_2) * normal_2), torch.sum(normal_1 * normal_2))

    def forward(self, input_dicts: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        positions = input_dicts["positions"].clone()
        cell = input_dicts["cell"].clone()
        input_dicts["side_array"].requires_grad_(False).to(torch.int64)
        cv_now = torch.stack(
            [
                self._torsion(positions, cell, self.phi_indices),
                self._torsion(positions, cell, self.psi_indices),
            ]
        ).flatten()
        input_dicts["cv_now"] = cv_now.to(dtype=positions.dtype)
        input_dicts["forces"] = torch.zeros_like(positions)
        input_dicts["virial"] = torch.zeros((9,), device=positions.device, dtype=positions.dtype)
        input_dicts["bias"] = torch.zeros((), device=positions.device, dtype=positions.dtype)
        return input_dicts


if __name__ == "__main__":
    output = Path(__file__).with_name("alanine-torsion-cvonly.pt")
    torch.jit.script(MatchingAlanineCVOnly().to(dtype=torch.double).eval()).save(str(output))
    print(f"Saved {output}")
