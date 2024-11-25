#pragma once
#include "potential.cuh"
#include <stdio.h>
#include <vector>
#include "neighbor.cuh"
#include "utilities/error.cuh"
#include "force/force.cuh"
#include "model/atom.cuh"
#include "utilities/read_file.cuh"


class TargetOpt: public Potential
{
public:
  int test = 9;

  TargetOpt();

  void parse_target_opt(const char** param, int num_param, Force& force);

  void compute(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom);
  
private:
  int step = 0;
  double k_end = 5.0;
  int tau = 100;
  int natoms;
  double core_size;
  double rc;
  double vert_part = 0.0;

  bool is_small_box = false;
  int i_group;
  Force* p_force;
  GPU_Vector<int> i_pick;
  int max_neighbor = 100;
  double k;
  GPU_Vector<int> NN_target; // neighbor number
  GPU_Vector<int> NL_target; // neighbor list
  GPU_Vector<double> dpos_target; // dpos corresponding to NL_target
};