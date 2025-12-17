#pragma once
#include "potential.cuh"
#include <stdio.h>
#include <vector>
#include "neighbor.cuh"
#include "utilities/error.cuh"
#include "force/force.cuh"
#include "model/atom.cuh"
#include "utilities/read_file.cuh"


class LJ_2d: public Potential
{
public:
  int test = 9;

  LJ_2d();

  void parse_lj_2d(const char** param, int num_param, Force& force);

  void compute(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom);
  
private:
  int itype;
  double mepsilon;
  double mw;
  double msigma;

  int step = 0;
  int natoms;
  Force* p_force;
};