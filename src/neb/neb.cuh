#pragma once
#include "force/force.cuh"
// #include "minimize/minimizer.cuh"
#include "minimize/minimizer_fire_jqh.cuh"
#include "utilities/common.cuh"
#include "utilities/read_file.cuh"
#include "model/read_xyz.cuh"
#include "model/atoms.cuh"
#include "model/box.cuh"
#include "model/atom.cuh"
#include <deque>
#include <cstring>
using namespace std;
#include "measure/dump_position.cuh"
#include "measure/parse_utilities.cuh"

class NEB: public BaseAtoms
{
private:
  double k = 0.1;

  // void one_neb_step();
  

public:
  int step = 0;
  bool climb = true;
  bool variable_cell = true;
  double pressure = 0.0;
  int max_steps = 0;
  double force_tolerance = 0.0;
  int minimizer_type = 0;
  deque<Atoms*> images;
  GPU_Vector<double> positions;
  Dump_Position dump_position;
  // Force force;

  NEB();

  NEB(Atoms atoms, const int number_of_atoms, const int number_of_steps, const double force_tolerance)
  {
  }

  void parse_neb(const char** param, int num_param, Force& force);

  void compute();

  GPU_Vector<double>& get_positions();

  // void set_positions();

  GPU_Vector<double>& get_forces();

  void norm_neb();

  void vcneb();

};

