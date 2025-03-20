#pragma once
#include "force/force.cuh"
#include "force/nep3.cuh"
// #include "minimize/minimizer.cuh"
#include "minimize/minimizer_fire_jqh.cuh"
#include "utilities/common.cuh"
#include "model/atoms.cuh"
#include "model/box.cuh"
#include "model/atom.cuh"
#include "measure/dump_position.cuh"
#include "measure/parse_utilities.cuh"
#include <algorithm>
#include <deque>
#include <list>
#include <map>
#include <cstring>
#include <cmath>
#include <cusolverDn.h>
#include <force/neighbor.cuh>

struct Spring
{
  double k, de, nt;
  GPU_Vector<double> t;
  
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

  BaseTangentMethod(double k0):k(k0) {};
    
  virtual GPU_Vector<double> compute_tangent(Spring& spring1, Spring& spring2) = 0;
  
  virtual void add_image_force(
    int size,
    double& tangential_force,
    double* tangent,
    Spring& spring1,
    Spring& spring2,
    double* imgforce) = 0;
};

class NormalTangentMethod: public BaseTangentMethod
{
public:
  NormalTangentMethod(){};

  NormalTangentMethod(double k0):BaseTangentMethod(k0) {};
  
  GPU_Vector<double> compute_tangent(Spring& spring1, Spring& spring2);

  void add_image_force(
    int size,
    double& tangential_force,
    double* tangent,
    Spring& spring1,
    Spring& spring2,
    double* imgforce);
};
class ImprovedTangentMethod: public BaseTangentMethod
{
public:
  ImprovedTangentMethod(){};

  ImprovedTangentMethod(double k0):BaseTangentMethod(k0) {};
  
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
  bool auto_k = false;
  std::vector<double> pressure = {0.0};
  bool has_mid = false;
  int n_interpolate = 0;
  bool need_relax = false;
  bool climb = false;
  bool remove_translation = true;
  bool remove_rotation = true;
  bool variable_cell = true;
  bool var_image_number = true;
  int vi_check_coord = 0; //  0: no check
  double vicc_num = 0.0; // >0 & <1: percent, >=1: number
  double vicc_rc = 1.7; 
  int vi_interval = 20;
  double vi_cell_factor = -1.0;
  double min_dist = 0.01, max_dist = 0.1;
  int dist_ncount = 10;
  int dump_interval = -1;
  int peek_interval = -1;
  int max_steps = 0;
  std::string istate_name = "is.xyz";
  std::string fstate_name = "fs.xyz";
  std::string mid_name = "mid.xyz";
  std::string traj_name = "";
  std::vector<std::string> mid_name_list;
  std::string tangent_method_name = "improved";


  // private variables
  // cublasHandle_t handle;
  std::vector<double> klist;
  std::unique_ptr<Minimizer> minimizer;
  std::vector<const char *> optimizer_opt;
  int imax;
  std::list<int> imins;
  std::list<int> imaxes;
  Dump_Position dump_position;
  // std::vector<pair<int,Atoms*>> mid_list;
  std::vector<int> imid_list; // the positions that each mid_image should be insert into
  std::vector<double> h_ref{9};
  double first_energy = 0.0;
  double last_energy = 0.0;
  int vi_count = 0;
  int step = 0;
  double force_tolerance;
  int minimizer_type;
  int nimages, natoms_per_image, n_realatoms;
  double optimize_factor;

  void find_min_max();

  void initialize_images();

  void initialize_compute();

  void check_dist();

public:
  std::vector<std::unique_ptr<Atoms>> images;
  std::vector<double> image_energies;
  BaseTangentMethod* tangentmethod;

  NEB();

  double get_energy();

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

  void write_energies();

  void interpolate();

};
