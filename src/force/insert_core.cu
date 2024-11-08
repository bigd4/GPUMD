#include "insert_core.cuh"

static __global__ void gpu_calc_spring_force()
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  apply_mic
}

void InsertCore::compute(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  calculate();
  pair_list
  find_neighbor(
      N1,
      N2,
      rc,
      box,
      type,
      position_per_atom,
      lj_data.cell_count,
      lj_data.cell_count_sum,
      lj_data.cell_contents,
      lj_data.NN,
      lj_data.NL);
      
  apply_mic(box, )
}

void find_neighbor(
  const int N1,
  const int N2,
  double rc,
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& cell_count,
  GPU_Vector<int>& cell_count_sum,
  GPU_Vector<int>& cell_contents,
  GPU_Vector<int>& NN,
  GPU_Vector<int>& NL)