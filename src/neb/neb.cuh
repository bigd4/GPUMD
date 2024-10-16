#pragma once
#include "force/force.cuh"
#include "minimize/minimizer.cuh"
#include "minimize/minimizer_fire.cuh"
#include "utilities/common.cuh"
#include "utilities/read_file.cuh"
#include "model/read_xyz.cuh"
#include "model/atoms.cuh"
#include "model/box.cuh"
#include "model/atom.cuh"
#include <deque>
#include <cstring>
using namespace std;

class NEB
{
private:
  double k = 0.1;
  

public:
  bool climb = true;
  bool variable_cell = true;
  deque<Atoms*> images;

  NEB();

  NEB(Atoms atoms, const int number_of_atoms, const int number_of_steps, const double force_tolerance)
  {
  }

  void compute(){

  };

  void parse_neb(
  const char** param,
  int num_param);
};

