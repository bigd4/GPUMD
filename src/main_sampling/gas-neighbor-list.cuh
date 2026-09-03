#pragma once

#include "force/neighbor.cuh"
#include <torch/torch.h>

namespace gas_neighbor {

// GAS sampling routes use cell edges of neighbor_rc/2, so every axis needs a
// +/-2 stencil.  Small periodic grids alias multiple offsets to the same cell;
// suppress those aliases to emit each directed atom pair exactly once.
static __global__ void find_neighbor_list_large_box(
  const float neighbor_rc,
  const int max_neighbors,
  const int n_atoms,
  const int nx,
  const int ny,
  const int nz,
  const Box box,
  const int* __restrict__ cell_count,
  const int* __restrict__ cell_count_sum,
  const int* __restrict__ cell_contents,
  const double* __restrict__ x,
  const double* __restrict__ y,
  const double* __restrict__ z,
  int* nn_radial,
  int* nl_radial)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 >= n_atoms) {
    return;
  }

  const double x1 = x[n1];
  const double y1 = y[n1];
  const double z1 = z[n1];
  int count = 0;
  int cell_id, cell_x, cell_y, cell_z;
  find_cell_id(
    box,
    x1,
    y1,
    z1,
    2.0f / neighbor_rc,
    nx,
    ny,
    nz,
    cell_x,
    cell_y,
    cell_z,
    cell_id);

  for (int dz = -2; dz <= 2; ++dz) {
    int neighbor_z = cell_z + dz;
    if (box.pbc_z) {
      neighbor_z = (neighbor_z % nz + nz) % nz;
      bool duplicate_z = false;
      for (int previous = -2; previous < dz; ++previous) {
        const int previous_z = ((cell_z + previous) % nz + nz) % nz;
        duplicate_z = duplicate_z || previous_z == neighbor_z;
      }
      if (duplicate_z) {
        continue;
      }
    } else if (neighbor_z < 0 || neighbor_z >= nz) {
      continue;
    }

    for (int dy = -2; dy <= 2; ++dy) {
      int neighbor_y = cell_y + dy;
      if (box.pbc_y) {
        neighbor_y = (neighbor_y % ny + ny) % ny;
        bool duplicate_y = false;
        for (int previous = -2; previous < dy; ++previous) {
          const int previous_y = ((cell_y + previous) % ny + ny) % ny;
          duplicate_y = duplicate_y || previous_y == neighbor_y;
        }
        if (duplicate_y) {
          continue;
        }
      } else if (neighbor_y < 0 || neighbor_y >= ny) {
        continue;
      }

      for (int dx = -2; dx <= 2; ++dx) {
        int neighbor_x = cell_x + dx;
        if (box.pbc_x) {
          neighbor_x = (neighbor_x % nx + nx) % nx;
          bool duplicate_x = false;
          for (int previous = -2; previous < dx; ++previous) {
            const int previous_x = ((cell_x + previous) % nx + nx) % nx;
            duplicate_x = duplicate_x || previous_x == neighbor_x;
          }
          if (duplicate_x) {
            continue;
          }
        } else if (neighbor_x < 0 || neighbor_x >= nx) {
          continue;
        }

        const int neighbor_cell =
          neighbor_z * nx * ny + neighbor_y * nx + neighbor_x;
        const int atoms_in_cell = cell_count[neighbor_cell];
        const int offset = cell_count_sum[neighbor_cell];
        for (int m = 0; m < atoms_in_cell; ++m) {
          const int n2 = cell_contents[offset + m];
          if (n1 == n2) {
            continue;
          }
          double dx12 = x[n2] - x1;
          double dy12 = y[n2] - y1;
          double dz12 = z[n2] - z1;
          apply_mic(box, dx12, dy12, dz12);
          const double distance2 = dx12 * dx12 + dy12 * dy12 + dz12 * dz12;
          if (distance2 < neighbor_rc * neighbor_rc) {
            if (count < max_neighbors) {
              nl_radial[count * n_atoms + n1] = n2;
            }
            ++count;
          }
        }
      }
    }
  }
  nn_radial[n1] = count;
}

inline torch::Tensor to_side_array(torch::Tensor& neighbor_list, int n_atoms)
{
  const int max_neighbors =
    static_cast<int>(neighbor_list.numel()) / n_atoms;
  auto starts = torch::arange(
    0,
    n_atoms,
    torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA));
  starts = starts.repeat({max_neighbors});
  const auto mask = neighbor_list >= 0;
  return torch::stack(
    {starts.masked_select(mask),
     neighbor_list.masked_select(mask).to(torch::kLong)},
    1);
}

} // namespace gas_neighbor
