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

struct Spring
{
  double k;
  double de;
  GPU_Vector<double> t;
  double nt;
  
  Spring(){};

  Spring(double k0, double de0, GPU_Vector<double> t0);
};

class BaseTangentMethod
{
protected:
  double k = 0.1;
  // double nt1, nt2;

public:
  BaseTangentMethod(){};

  BaseTangentMethod(double k0)
  :k(k0) {};

  // virtual void compute_tangent(
  //   GPU_Vector<double>& tangent,
  //   GPU_Vector<double>& t1,
  //   GPU_Vector<double>& t2,
  //   double de1,
  //   double de2) = 0;

  // virtual void add_image_force(
  //   int size,
  //   double& tangential_force,
  //   double* tangent,
  //   double* imgforce) = 0;
    
  virtual GPU_Vector<double> compute_tangent(Spring& spring1, Spring& spring2) = 0;
  
  virtual void add_image_force(
    int size,
    double& tangential_force,
    double* tangent,
    Spring& spring1,
    Spring& spring2,
    double* imgforce) = 0;

};

class ImprovedTangentMethod: public BaseTangentMethod
{
public:
  ImprovedTangentMethod(){};

  ImprovedTangentMethod(double k0)
  :BaseTangentMethod(k0) {};
  
  // void compute_tangent(
  //   GPU_Vector<double>& tangent,
  //   GPU_Vector<double>& t1,
  //   GPU_Vector<double>& t2,
  //   double de1,
  //   double de2);

  // void add_image_force(
  //   int size,
  //   double& tangential_force,
  //   double* tangent,
  //   double* imgforce);
  
  GPU_Vector<double> compute_tangent(Spring& spring1, Spring& spring2);

  void add_image_force(
    int size,
    double& tangential_force,
    double* tangent,
    Spring& spring1,
    Spring& spring2,
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
  int vi_interval = 20;
  double min_dist = 0.005, max_dist = 0.1;
  int dump_interval = -1;
  int peek_interval = -1;
  int max_steps = 0;
  string istate_name = "is.xyz";
  string fstate_name = "fs.xyz";
  string mid_name = "mid.xyz";
  vector<string> mid_name_list;


  // private variables
  // cublasHandle_t handle;
  unique_ptr<Minimizer> minimizer;
  vector<const char *>optimizer_opt;
  list<int> imaxes;
  Dump_Position dump_position;
  vector<pair<int,Atoms*>> mid_list;
  vector<double> ref_h{9};
  double first_energy = 0.0;
  double last_energy = 0.0;
  int vi_count = 0;
  int step = 0;
  double force_tolerance;
  int minimizer_type;
  int nimages, natoms_per_image;
  double optimize_factor;


  void find_min_max();

  void initialize_compute();

  void check_dist();

public:
  deque<Atoms*> images;
  vector<double> image_energies;
  ImprovedTangentMethod tangentmethod;


  NEB();

  void parse_options(const char** param, int num_param, int& n);

  // NEB(Atoms atoms, const int number_of_atoms, const int number_of_steps, const double force_tolerance)
  // {}

  void parse_neb(const char** param, int num_param, Force& force);

  void reset_minimizer(int number_of_atoms, int max_steps, double force_tolerance);

  void compute();

  GPU_Vector<double>& build_positions();

  void set_positions();

  // GPU_Vector<double>& get_forces();

  void run_neb();

  void write_neb_traj(const char* filename, const char* mode);

  void interpolate();

  // void vcneb();

};
