#pragma once
#include "potential.cuh"
#include <stdio.h>
#include <vector>
#include "neighbor.cuh"
#include "utilities/error.cuh"


class InsertCore: public Potential
{
public:
  double core_size;
  int max_neighbor;
  double rc;
  double k;


  InsertCore(int natoms);

  void compute(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom);
  
  void calc_spring_force();
};