#pragma once

#include <memory>
#include <string>
#include <torch/script.h>
#include <torch/torch.h>
#ifdef GAS_HAS_AOTI
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#endif
#include <unordered_map>
#include <vector>

class GASModelRunner
{
public:
  GASModelRunner(
    const std::string& model_path,
    const std::string& expected_route,
    int n_atoms,
    int cv_size,
    int max_edges,
    int max_history,
    int opes_capacity = 0);

  torch::Dict<std::string, torch::Tensor>
  run(const torch::Dict<std::string, torch::Tensor>& inputs);

  torch::Dict<std::string, torch::Tensor>
  initialize_opes_state(int capacity);

  bool is_aoti() const { return backend_ == "aoti"; }
  const std::string& backend() const { return backend_; }
  const std::string& route() const { return route_; }
  const std::unordered_map<std::string, std::string>& metadata() const
  {
    return metadata_;
  }

private:
  void validate_package(
    const std::string& model_path,
    const std::string& expected_route,
    int n_atoms,
    int cv_size,
    int max_edges,
    int max_history,
    int opes_capacity);
  void validate_tensor(
    const std::string& name,
    const torch::Tensor& value) const;
  torch::Tensor input_or_cvonly_default(
    const torch::Dict<std::string, torch::Tensor>& inputs,
    const std::string& name) const;

  std::string backend_ = "torchscript";
  std::string route_;
  int cv_size_ = 0;
  int max_history_ = 0;
  std::unique_ptr<torch::jit::script::Module> torchscript_;
#ifdef GAS_HAS_AOTI
  std::unique_ptr<torch::inductor::AOTIModelPackageLoader> aoti_;
#endif
  std::unordered_map<std::string, std::string> metadata_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
};
