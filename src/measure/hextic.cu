/*
    Copyright 2017 Zheyong Fan, Ville Vierimaa, Mikko Ervasti, and Ari Harju
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/
#include "hextic.cuh"
#include "force/neighbor.cuh"
#include "utilities/read_file.cuh"
#include "model/box.cuh"

const int MN_nearest = 15; // maximum number of neighbors for each atom

namespace
{
void __global__ get_cos_sin(
  const int N, Box box,
  int* NN, int* NL, 
  double* sum_cos, double* sum_sin, double* phi,
  const double* __restrict__ x,
  const double* __restrict__ y)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    double x1 = x[n1];
    double y1 = y[n1];
    sum_cos[n1] = 0.0;
    sum_sin[n1] = 0.0;
    phi[n1] = 0.0;
    for (int i = 0; i < NN[n1]; i++) {
      int n2 = NL[n1 + N * i];
      double x12 = x[n2] - x1;
      double y12 = y[n2] - y1;
      double tmpz = 0.;
      apply_mic(box, x12, y12, tmpz);
      double norm = sqrt(x12 * x12 + y12 * y12 ) * NN[n1];
      sum_cos[n1] += x12 / norm;
      sum_sin[n1] += y12 / norm;
    }
    phi[n1] = sum_cos[n1] * sum_cos[n1] + sum_sin[n1] * sum_sin[n1];
  }
}

void __global__ get_orien_order(
  const int N, Box box, int mesh,
  int* order_count, double* orien_order, 
  double* radius,
  double* sum_cos, double* sum_sin,
  const double* __restrict__ x,
  const double* __restrict__ y)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 < N) {
    const double x1 = x[n1];
    const double y1 = y[n1];
    for (int n2 = 0; n2 < N;  n2++) {
      double x12 = x[n2] - x1;
      double y12 = y[n2] - y1;
      double tmpz = 0.;
      apply_mic(box, x12, y12, tmpz);
      const double distance_square = x12 * x12 + y12 * y12;
      for (int j = 0; j < mesh; j++) {
        if ((distance_square > (radius[j] * radius[j])) && (distance_square < (radius[j + 1] * radius[j + 1]))) {
          const double double_p = sum_cos[n1] * sum_cos[n2] + sum_sin[n1] * sum_sin[n2];
          orien_order[n1 * mesh + j] += double_p ;
          order_count[n1 * mesh + j] += 1;
        }
      }
    }
  }
}

} // namespace

void Hextic::parse(const char** param, const int num_param)
{
  printf("Compute hextic order parameters.\n");
  compute_ = true;

  if (num_param != 5) {
    PRINT_INPUT_ERROR("compute_hextic should have 4 parameters.\n");
  }
  if (!is_valid_int(param[1], &sample_interval_)) {
    PRINT_INPUT_ERROR("sample interval should be an integer.\n");
  }
  if (sample_interval_ <= 0) {
    PRINT_INPUT_ERROR("sample interval should be positive.\n");
  }

  if (!is_valid_real(param[2], &cutoff_)) {
    PRINT_INPUT_ERROR("sample interval should be an integer.\n");
  }
  if (cutoff_ <= 0) {
    PRINT_INPUT_ERROR("sample interval should be positive.\n");
  }

  if (!is_valid_real(param[3], &correlation_length_)) {
    PRINT_INPUT_ERROR("sample interval should be an integer.\n");
  }
  if (correlation_length_ <= 0) {
    PRINT_INPUT_ERROR("sample interval should be positive.\n");
  }

  if (!is_valid_int(param[4], &mesh_)) {
    PRINT_INPUT_ERROR("sample interval should be an integer.\n");
  }
  if (mesh_ <= 0) {
    PRINT_INPUT_ERROR("sample interval should be positive.\n");
  }
  printf("    sample interval is %d.\n    cutoff is %g.\n"
         "    correlation_length is %g.\n    mesh is %d.\n", 
         sample_interval_, cutoff_, correlation_length_, mesh_);
  const double step = correlation_length_ / mesh_;
  double radius[mesh_ + 1];
  for (int i = 0; i < mesh_ + 1; i++){
    radius[i] = i * step;
  }
  radius_.resize(mesh_ + 1);
  radius_.copy_from_host(radius);
}

void Hextic::preprocess(
    const int num_atoms)
{
  if (!compute_)
    return;
  initialize_parameters(num_atoms);
  allocate_memory();
}


void Hextic::process(
  const int step,
  Box& box,
  GPU_Vector<int>& type,
  GPU_Vector<double>& position_per_atom,
  std::vector<double>& cpu_position_per_atom)
{
  if (!compute_)
    return;
  if ((step + 1) % sample_interval_ != 0)
    return;
  int block_size = 128;
  int grid_size = (num_atoms_ - 1) / block_size + 1;
  const double* x = position_per_atom.data();
  const double* y = position_per_atom.data() + num_atoms_;
  find_neighbor(
    0, num_atoms_, cutoff_, box, type, position_per_atom, 
    cell_count, cell_count_sum, cell_contents, NN, NL);
  get_cos_sin<<<grid_size, block_size>>>(
    num_atoms_, box, NN.data(), NL.data(), sum_cos.data(), sum_sin.data(), phi.data(), x, y);
  CUDA_CHECK_KERNEL
  get_orien_order<<<grid_size, block_size>>>(
    num_atoms_, box, mesh_, order_count.data(), orien_order.data(), radius_.data(), sum_cos.data(), sum_sin.data(), x, y);
  output_phi(cpu_position_per_atom);
  output_order_correlation();
}

void Hextic::postprocess()
{
  if (!compute_)
    return;

  CHECK(cudaDeviceSynchronize()); // needed for pre-Pascal GPU
  compute_ = false;
}

void Hextic::initialize_parameters(
  const int num_atoms)
{
  num_atoms_ = num_atoms;
}

void Hextic::allocate_memory()
{
  NN.resize(num_atoms_);
  NL.resize(num_atoms_ * MN_nearest);
  cell_count.resize(num_atoms_);
  cell_count_sum.resize(num_atoms_);
  cell_contents.resize(num_atoms_);
  order_count.resize(num_atoms_ * mesh_);
  orien_order.resize(num_atoms_ * mesh_);
  cpu_order_count.resize(num_atoms_ * mesh_);
  cpu_orien_order.resize(num_atoms_ * mesh_);
  sum_cos.resize(num_atoms_);
  sum_sin.resize(num_atoms_);
  phi.resize(num_atoms_);
  cpu_cos.resize(num_atoms_);
  cpu_sin.resize(num_atoms_);
  cpu_phi.resize(num_atoms_);
}

void Hextic::output_phi(
  std::vector<double>& cpu_position_per_atom)
{
  FILE* fid_dos = fopen("hehe.out", "a");
  sum_cos.copy_to_host(cpu_cos.data());
  sum_sin.copy_to_host(cpu_sin.data());
  phi.copy_to_host(cpu_phi.data());
  for (int n = 0; n < num_atoms_; n++){
    fprintf(
      fid_dos, "%g %g %g %g %g %g\n", cpu_position_per_atom[n], cpu_position_per_atom[n + num_atoms_], 
      cpu_position_per_atom[n + 2 * num_atoms_], cpu_cos[n], cpu_sin[n], cpu_phi[n]);
  }
  fflush(fid_dos);
  fclose(fid_dos);
}

void Hextic::output_order_correlation()
{
  FILE* fid_dos = fopen("haha.out", "a");
  order_count.copy_to_host(cpu_order_count.data());
  orien_order.copy_to_host(cpu_orien_order.data());
  for (int m = 0; m < mesh_; m++)
  {
    int tmp1 = 0;
    double tmp2 = 0.0;
    for (int n = 0; n < num_atoms_; n++){
      tmp1 += cpu_order_count[n * mesh_ + m];
      tmp2 += cpu_orien_order[n * mesh_ + m];
    }
    double tmp3 = 0.;
    if (tmp1 > 0) tmp3 = tmp2 / tmp1;
    fprintf(fid_dos, "%g ", tmp3);
  }
  fprintf(fid_dos, "\n");
  fflush(fid_dos);
  fclose(fid_dos);
}
