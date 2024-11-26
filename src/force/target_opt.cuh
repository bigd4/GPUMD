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
  double rc = 2.0;
  double k_end = 5.0;
  int tau = 100;
  double vert_part = 0.0;
  double core_size;
  int max_neighbor = 100;

  int step = 0;
  int natoms;
  bool is_small_box = false;
  int i_group;
  Force* p_force;
  double k;
  GPU_Vector<int> NN_target; // neighbor number
  GPU_Vector<int> NL_target; // neighbor list
  std::vector<GPU_Vector<int>> i_pick_list;
  std::vector<GPU_Vector<double>> dpos_target_list; // dpos corresponding to NL_target
};