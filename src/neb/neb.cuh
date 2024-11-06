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
#include <map>
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

  void add_image_force(
    int size,
    double& tangential_force,
    double* tangent,
    double* imgforce);
};

class NEB: public BaseAtoms
{
private:
  // compute setting
  double k = 0.1;
  double pressure = 0.0;
  bool has_mid = false;
  int n_interpolate = 3;
  bool need_relax = false;
  bool climb = false;
  bool remove_transition = true;
  bool variable_cell = true;
  bool var_image_number = true;
  int vi_interval = 10;
  double min_dist = 0.005, max_dist = 0.1;
  string istate_name = "is.xyz";
  string fstate_name = "fs.xyz";
  string mid_name = "mid.xyz";
  double optimize_factor;


  cublasHandle_t handle;
  double first_energy = 0.0;
  double last_energy = 0.0;
  int vi_count = 0;
  vector<double> ref_h = vector<double>(9);
  unique_ptr<Minimizer> minimizer;
  list<int> imaxes;
  Dump_Position dump_position;
  int dump_interval = -1;
  int step = 0;
  int max_steps = 0;
  double force_tolerance = 0.0;
  int minimizer_type = 0;
  int nimages, natoms_per_image;
  map<int,Atoms*> mid_list;


  void find_min_max();

  void initialize_compute();

  void check_dist();

public:
  deque<Atoms*> images;
  vector<double> image_energies;
  ImprovedTangentMethod tangentmethod;


  NEB();

  void parse_options(const char** param, int num_param, int& n);

  NEB(Atoms atoms, const int number_of_atoms, const int number_of_steps, const double force_tolerance)
  {
  }

  void parse_neb(const char** param, int num_param, Force& force);

  void reset_minimizer(int number_of_atoms, int max_steps, double force_tolerance);

  void compute();

  GPU_Vector<double>& build_positions();

  void set_positions();

  // GPU_Vector<double>& get_forces();

  void run_neb();

  void write_neb_traj();

  void interpolate(int n);

  // void vcneb();

};
