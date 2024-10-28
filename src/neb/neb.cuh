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
#include <list>
#include <cstring>
#include <cmath>
using namespace std;
#include "measure/dump_position.cuh"
#include "measure/parse_utilities.cuh"

class ImprovedTangentMethod
{
private:
  double k = 0.1;
  double nt1, nt2, scale1, scale2;
  cublasHandle_t handle;

public:
  ImprovedTangentMethod(){};

  ImprovedTangentMethod(cublasHandle_t& handle0, double k0)
  :handle(handle0), k(k0) {};
  
  void compute_tangent(
    GPU_Vector<double>& tangent,
    GPU_Vector<double>& t1,
    GPU_Vector<double>& t2,
    double de1,
    double de2);

  void compute_image_force(
    GPU_Vector<double>& tangential_force,
    GPU_Vector<double>& tangent,
    GPU_Vector<double>& imgforce);
};

class NEB: public BaseAtoms
{
private:
  double k = 0.1;
  cublasHandle_t handle;
  double first_energy = 0.0;
  double last_energy = 0.0;

  void find_min_max();


  void initialize();

public:
  int nimages;
  int natoms_per_image;
  int step = 0;
  bool climb = true;
  bool variable_cell = true;
  // double pressure = 0.0;
  list<int> imaxes;
  int max_steps = 0;
  double force_tolerance = 0.0;
  int minimizer_type = 0;
  deque<Atoms*> images;
  vector<double> image_energies;
  Dump_Position dump_position;
  // Force force;
  ImprovedTangentMethod tangentmethod;


  NEB();

  NEB(Atoms atoms, const int number_of_atoms, const int number_of_steps, const double force_tolerance)
  {
  }

  void parse_neb(const char** param, int num_param, Force& force);


  void compute();

  GPU_Vector<double>& build_positions();

  void set_positions();

  // GPU_Vector<double>& get_forces();

  void norm_neb();

  void vcneb();

  template<class T> void run_neb();

};

template<class T>
void NEB::run_neb(){
  T image=*images[0];
}
