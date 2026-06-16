#pragma once
#include "potential.cuh"
#include <stdio.h>
#include <vector>
#include "neighbor.cuh"
#include "utilities/error.cuh"
#include "force/force.cuh"
#include "model/atom.cuh"
#include "utilities/read_file.cuh"


class Morse: public Potential
{
public:

  Morse();

  void parse_morse(const char** param, int num_param, Force& force);

  void compute(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom);
  
private:
  int itype;
  double md;
  double ma;
  double mz;
  double mw;
  double mk0;

  int step = 0;
  int natoms;
  Force* p_force;
};