#pragma once
#include "force/force.cuh"
#include "force/nep.cuh"
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
#include <utility>
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
  ImprovedTangentMethod(double k0):BaseTangentMethod(k0) {};

  GPU_Vector<double> compute_tangent (Spring& spring1, Spring& spring2) override;

  void add_image_force(
    int size,
    double& tangential_force,
    double* tangent,
    Spring& spring1,
    Spring& spring2,
    double* imgforce) override;
};

class ModifiedImprovedTangentMethod: public ImprovedTangentMethod
{
public:  
  // ---- workspace vectors ----
  GPU_Vector<double> perp_force;
  GPU_Vector<double> unit_perp_force;
  GPU_Vector<double> ori_spring_force;
  GPU_Vector<double> par_spring_force;
  GPU_Vector<double> perp_spring_force;
  GPU_Vector<double> dneb_force;
  int workspace_size = 0;

  ModifiedImprovedTangentMethod(double k0):ImprovedTangentMethod(k0) {};

  void ensure_workspace(int size);


  void add_image_force(
    int size,
    double& tangential_force,
    double* tangent,
    Spring& spring1,
    Spring& spring2,
    double* imgforce) override;
};


class NEB: public BaseAtoms
{
private:
  // compute setting
  double k = 0.1;
  bool energy_based_spacing = false;
  std::vector<double> pressure = {0.0};
  bool has_mid = false;
  int n_interpolate = 0;
  bool need_relax = false;
  bool climb = false;
  bool find_min = false;
  bool dynamic_relaxation = false;
  double scale_fmax = 0.0;
  double dyneb_energy_exponent = 1.0;
  double dyneb_peak_width = 0.1;
  double etol = 0.0;
  double trim_etol = 0.0;
  bool has_trim_etol = false;
  double energy_spacing_damping = 0.1;
  double energy_spacing_strength = 0.8;
  double energy_spacing_exponent = 1;
  double energy_spacing_dist_power = 0.5;
  bool remove_translation = true;
  bool remove_rotation = true;
  bool variable_cell = true;
  bool match_cell_to_initial = false;
  bool find_mic = false;

  bool image_number_adjustment = true;
  bool trim_images = false;
  double trim_similar_tol = 0.01;
  bool ina_k = false;
  double ina_k_efficient = 1.8;
  double ina_insert_midpoint_weight = 0.0;
  double cell_metric_active_threshold = 0.1;
  int ina_check_coord = 0; //  0: no check
  double inacc_num = 0.0; // >0 & <1: percent, >=1: number
  double inacc_rc = 1.7;
  int ina_interval = 20;
  int ina_local_relax_steps = 0;
  int ina_local_relax_neighbors = 1;
  double cell_factor = -1.0;
  std::vector<std::pair<int, double>> ina_force_tol_stages;
  double min_dist = 0.01, max_dist = 0.1;
  int dist_ncount = 10;
  bool print_k = false;
  int print_interval = 1;
  int diagnostic_interval = 0;
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
  std::vector<double> energy_spacing_factor;
  std::vector<double> k_effective_list;
  std::vector<double> kori_list;
  std::unique_ptr<Minimizer> minimizer;
  std::vector<const char *> optimizer_opt;
  int imax;
  std::list<int> imins;
  std::list<int> imaxes;
  double fmax;
  // std::vector<pair<int,Atoms*>> mid_list;
  std::vector<int> imid_list; // the positions that each mid_image should be insert into
  std::vector<double> h_ref{9};
  double first_energy = 0.0;
  double last_energy = 0.0;
  int ina_count = 0;
  int step = 0;
  bool count_force_calc = false;
  int n_force_calc = 0;
  double force_tolerance;
  double cell_metric_scale_default = 1.0;
  double minimizer_cell_metric_scale = 1.0;
  int cell_metric_active_atoms = 0;
  int minimizer_type;
  int nimages, natoms_per_image, n_realatoms;
  std::vector<char> dyneb_active;
  GPU_Vector<double> dyneb_force_max;
  GPU_Vector<double> dyneb_position_delta;
  int ina_local_relax_remaining = 0;
  bool ina_local_tracking = false;
  bool ina_local_changed = false;
  std::vector<char> ina_local_active;

  void find_min_max(double etol=0.0);

  void initialize_images();

  void prepare_fixed_cell_images();

  void align_images_by_mic();

  double estimate_active_atom_scale(Atoms& initial_atoms, Atoms& final_atoms);

  double estimate_cell_metric_scale();

  void initialize_compute();

  void apply_dynamic_relaxation();

  void apply_ina_local_relaxation();

  void begin_ina_local_tracking();

  void mark_ina_local_region(int image_index);

  void notify_ina_image_inserted(int image_index);

  void notify_ina_image_erased(int image_index);

  void finish_ina_local_tracking();

  bool satisfy_ina_force_tolerence() const;

  void adjust_image_spacing(bool allow_remove, bool bootstrap);

  void adjust_image_number();

  bool update_minimizer_force_max(double force_max) override;

  void print_info(double force_max);

  void report_minimizer_state(
    double dt, double power, double alpha, int n_positive, bool reset) override;

  void report_imagewise_minimizer_state(
    const std::vector<double>& dt,
    const std::vector<double>& power,
    const std::vector<double>& alpha,
    const std::vector<int>& n_positive,
    const std::vector<int>& reset) override;

public:
  std::vector<std::unique_ptr<Atoms>> images;
  std::vector<double> image_energies;
  std::unique_ptr<BaseTangentMethod> tangentmethod;

  NEB();

  ~NEB();

  double get_energy() override;

  void parse_options(const char** param, int num_param, int& n);

  // NEB(Atoms atoms, const int number_of_atoms, const int number_of_steps, const double force_tolerance)
  // {}

  void parse_neb(const char** param, int num_param, Force& force);

  void reset_minimizer(
    int number_of_atoms, int max_steps, double force_tolerance, bool print_flag=false);

  void compute() override;

  GPU_Vector<double>& build_positions();

  void set_positions();

  bool has_cell_degrees_of_freedom() const override { return variable_cell; }

  int get_real_atom_count_per_block() const override
  {
    return variable_cell ? n_realatoms : natoms_per_image;
  }

  int get_atoms_per_block() const override { return natoms_per_image; }

  // GPU_Vector<double>& get_forces();

  void run_neb();

  void write_neb_traj(const char* filename, const char* mode);

  void write_energies();

  void interpolate();

};
