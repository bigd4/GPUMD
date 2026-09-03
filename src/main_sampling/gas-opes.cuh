#pragma once

#include "force/neighbor.cuh"
#include "force/potential.cuh"
#include "gas-metad.cuh"
#include "gas-model-runner.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_vector.cuh"
#include <torch/script.h>
#include <torch/torch.h>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

struct OPESConfig
{
  Config common;
  int max_kernels = 4096;
  int state_interval = 50000;
  std::string state_path = "OPES_STATE.pt";
  std::string kernels_path = "OPES_KERNELS";
  bool stop_on_status = false;
  int stop_status = 0; // zero means any non-zero model status

  static OPESConfig from_file(const std::string& path, int n_atoms);
  void print() const;
};

class TorchOPES : public Potential
{
public:
  TorchOPES(const std::string& model_path, const std::string& cfg_path, int n_atoms);
  TorchOPES(
    const std::string& model_path,
    const std::string& cfg_path,
    const std::string& state_path,
    int n_atoms);
  ~TorchOPES() override;

  void compute(
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& positions,
    GPU_Vector<double>& potentials,
    GPU_Vector<double>& forces,
    GPU_Vector<double>& virial) override;

  bool should_stop() const;
  int current_status() const { return status_; }
  void checkpoint();

  static std::unique_ptr<TorchOPES>
  parse_OPES(const char** param, int num_param, int number_of_atoms);

private:
  void initialize(const std::string& model_path, const std::string& cfg_path);
  void initialize_state();
  void load_state(const std::string& state_path);
  void update_state(const torch::Dict<std::string, torch::Tensor>& output);
  void compute_large_box(Box& box, const GPU_Vector<double>& positions);
  void get_neighbor_list(Box& box, const GPU_Vector<double>& positions);
  void box_to_tri(Box& box);
  void log_colvar(const torch::Dict<std::string, torch::Tensor>& output, int evaluated_step);
  void log_kernel(const torch::Dict<std::string, torch::Tensor>& output, int evaluated_step);
  torch::Tensor from_cuda_memory(double* array, int size);

  int n_atoms_ = 0;
  int now_step_ = 0;
  int status_ = 0;
  bool initialized_ = false;
  bool state_dirty_ = false;
  bool restart_force_refresh_pending_ = false; // suppress exactly one pre-loop state update
  OPESConfig config_;
  std::unique_ptr<GASModelRunner> model_runner_;
  std::unordered_map<std::string, torch::Tensor> state_;

  std::vector<double> cpu_box_;
  GPU_Vector<int> cell_count_;
  GPU_Vector<int> cell_count_sum_;
  GPU_Vector<int> cell_contents_;
  GPU_Vector<int> nn_radial_;
  GPU_Vector<int> nl_radial_;
};
