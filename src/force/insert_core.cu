#include "insert_core.cuh"

// static __global__ void gpu_calc_spring_force(Box& box, )
// {
//   int n = blockIdx.x * blockDim.x + threadIdx.x;
//   apply_mic(box,);
// }

InsertCore::InsertCore(int natoms)
{
  max_neighbor = 100;

}

void InsertCore::compute(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  calc_spring_force();
  GPU_Vector<int> cell_count;
  GPU_Vector<int> cell_count_sum;
  GPU_Vector<int> cell_contents;
  GPU_Vector<int> NN; // neighbor number
  GPU_Vector<int> NL; // neighbor list
  find_neighbor(
      N1,
      N2,
      rc,
      box,
      type,
      position_per_atom,
      cell_count,
      cell_count_sum,
      cell_contents,
      NN,
      NL);
      
  // apply_mic(box, )
}

void InsertCore::calc_spring_force()
{
}

// void find_neighbor(
//   const int N1,
//   const int N2,
//   double rc,
//   Box& box,
//   const GPU_Vector<int>& type,
//   const GPU_Vector<double>& position_per_atom,
//   GPU_Vector<int>& cell_count,
//   GPU_Vector<int>& cell_count_sum,
//   GPU_Vector<int>& cell_contents,
//   GPU_Vector<int>& NN,
//   GPU_Vector<int>& NL){

//   }