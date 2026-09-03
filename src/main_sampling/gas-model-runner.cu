#include "gas-model-runner.cuh"

#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <torch/version.h>

namespace
{
bool has_suffix(const std::string& value, const std::string& suffix)
{
  return value.size() >= suffix.size() &&
    value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::vector<std::string> split_csv(const std::string& value)
{
  std::vector<std::string> result;
  std::stringstream stream(value);
  std::string item;
  while (std::getline(stream, item, ',')) {
    if (!item.empty()) {
      result.push_back(item);
    }
  }
  return result;
}

std::string required_metadata(
  const std::unordered_map<std::string, std::string>& metadata,
  const std::string& key)
{
  const auto found = metadata.find(key);
  if (found == metadata.end() || found->second.empty()) {
    throw std::runtime_error("AOTI package metadata is missing " + key);
  }
  return found->second;
}

long metadata_integer(
  const std::unordered_map<std::string, std::string>& metadata,
  const std::string& key)
{
  const std::string value = required_metadata(metadata, key);
  try {
    return std::stol(value);
  } catch (const std::exception&) {
    throw std::runtime_error("AOTI package metadata " + key + " is not an integer: " + value);
  }
}

std::string manifest_path_for(const std::string& model_path)
{
  return model_path.substr(0, model_path.size() - 4) + ".manifest.json";
}

std::string read_text(const std::string& path)
{
  std::ifstream input(path);
  if (!input.is_open()) {
    throw std::runtime_error("AOTI package requires its external manifest: " + path);
  }
  std::ostringstream buffer;
  buffer << input.rdbuf();
  return buffer.str();
}

std::string json_string(const std::string& text, const std::string& key)
{
  const std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*\\\"([^\\\"]*)\\\"");
  std::smatch match;
  if (!std::regex_search(text, match, pattern)) {
    throw std::runtime_error("Artifact manifest is missing string key " + key);
  }
  return match[1].str();
}

long json_integer(const std::string& text, const std::string& key)
{
  const std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*(-?[0-9]+)");
  std::smatch match;
  if (!std::regex_search(text, match, pattern)) {
    throw std::runtime_error("Artifact manifest is missing integer key " + key);
  }
  return std::stol(match[1].str());
}

void require_equal(
  const std::string& label,
  const std::string& package_value,
  const std::string& manifest_value)
{
  if (package_value != manifest_value) {
    throw std::runtime_error(
      "AOTI package/manifest mismatch for " + label + ": " + package_value +
      " != " + manifest_value);
  }
}

bool route_matches(const std::string& expected, const std::string& actual)
{
  if (expected == "bias_or_cvonly") {
    return actual == "bias" || actual == "cvonly";
  }
  return expected == actual;
}

std::string base_version(const std::string& version)
{
  const size_t suffix = version.find('+');
  return version.substr(0, suffix);
}

std::string compiled_cuda_version()
{
  const int major = CUDART_VERSION / 1000;
  const int minor = (CUDART_VERSION % 1000) / 10;
  return std::to_string(major) + "." + std::to_string(minor);
}

std::string compiled_cxx11_abi()
{
#ifdef _GLIBCXX_USE_CXX11_ABI
  return std::to_string(_GLIBCXX_USE_CXX11_ABI);
#else
  return "unknown";
#endif
}

std::vector<double> parse_number_array(const std::string& value, const std::string& key)
{
  if (value.size() < 2 || value.front() != '[' || value.back() != ']') {
    throw std::runtime_error("AOTI metadata " + key + " is not a numeric array");
  }
  std::vector<double> result;
  std::stringstream stream(value.substr(1, value.size() - 2));
  std::string item;
  while (std::getline(stream, item, ',')) {
    if (item.empty()) {
      continue;
    }
    try {
      result.push_back(std::stod(item));
    } catch (const std::exception&) {
      throw std::runtime_error("AOTI metadata " + key + " contains a non-number");
    }
  }
  return result;
}
} // namespace

GASModelRunner::GASModelRunner(
  const std::string& model_path,
  const std::string& expected_route,
  int n_atoms,
  int cv_size,
  int max_edges,
  int max_history,
  int opes_capacity)
  : cv_size_(cv_size), max_history_(max_history)
{
  if (has_suffix(model_path, ".pt2")) {
#ifndef GAS_HAS_AOTI
    throw std::runtime_error(
      "This GPU-Sampling executable was built without the AOTI backend. "
      "Use the matching .pt TorchScript artifact, or rebuild with LibTorch 2.13.0 "
      "and ENABLE_GAS_AOTI=ON.");
#else
    backend_ = "aoti";
    aoti_ = std::make_unique<torch::inductor::AOTIModelPackageLoader>(model_path);
    metadata_ = aoti_->get_metadata();
    if (cv_size_ <= 0) {
      cv_size_ = static_cast<int>(metadata_integer(metadata_, "pygas.cv_size"));
    }
    validate_package(
      model_path,
      expected_route,
      n_atoms,
      cv_size,
      max_edges,
      max_history,
      opes_capacity);
    input_names_ = split_csv(required_metadata(metadata_, "pygas.input_names"));
    output_names_ = split_csv(required_metadata(metadata_, "pygas.output_names"));
#endif
  } else if (has_suffix(model_path, ".pt")) {
    backend_ = "torchscript";
    torch::jit::setGraphExecutorOptimize(true);
    torchscript_ = std::make_unique<torch::jit::script::Module>(
      torch::jit::load(model_path, torch::kCUDA));
    torchscript_->eval();
    route_ = expected_route;
  } else {
    throw std::runtime_error("GAS model must end in .pt or .pt2: " + model_path);
  }
}

void GASModelRunner::validate_package(
  const std::string& model_path,
  const std::string& expected_route,
  int n_atoms,
  int cv_size,
  int max_edges,
  int max_history,
  int opes_capacity)
{
  require_equal("schema", required_metadata(metadata_, "pygas.schema"), "pygas-artifact-v1");
  require_equal("ABI", required_metadata(metadata_, "pygas.abi_version"), "1");
  route_ = required_metadata(metadata_, "pygas.route");
  if (!route_matches(expected_route, route_)) {
    throw std::runtime_error(
      "AOTI route mismatch: runtime expects " + expected_route + ", package is " + route_);
  }
  const std::string torch_version = required_metadata(metadata_, "pygas.torch_version");
  if (base_version(torch_version) != TORCH_VERSION || std::string(TORCH_VERSION) != "2.13.0") {
    throw std::runtime_error(
      "AOTI needs exact PyTorch/LibTorch 2.13.0; package=" + torch_version +
      ", LibTorch=" + TORCH_VERSION);
  }
  require_equal(
    "CUDA toolkit",
    required_metadata(metadata_, "pygas.cuda_version"),
    compiled_cuda_version());
  require_equal(
    "C++11 ABI",
    required_metadata(metadata_, "pygas.cxx11_abi"),
    compiled_cxx11_abi());
  cudaDeviceProp properties{};
  int device = 0;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&properties, device);
  const std::string capability =
    std::to_string(properties.major) + "." + std::to_string(properties.minor);
  require_equal(
    "compute capability",
    required_metadata(metadata_, "pygas.compute_capability"),
    capability);
  if (metadata_integer(metadata_, "pygas.n_atoms") != n_atoms ||
      (cv_size > 0 && metadata_integer(metadata_, "pygas.cv_size") != cv_size)) {
    throw std::runtime_error("AOTI atom/CV dimensions do not match GAScfg/model.xyz");
  }
  if (metadata_integer(metadata_, "pygas.max_edges") != max_edges) {
    throw std::runtime_error("AOTI max_edges does not match n_atoms * max_neighbors");
  }
  if ((route_ == "bias" || route_ == "cvonly") && max_history > 0 &&
      metadata_integer(metadata_, "pygas.max_cv_nums") != max_history) {
    throw std::runtime_error("AOTI history capacity does not match GAScfg max_cv_nums");
  }
  if (route_ == "opes" &&
      metadata_integer(metadata_, "pygas.opes_kernel_capacity") != opes_capacity) {
    throw std::runtime_error("AOTI OPES capacity does not match GAScfg opes_max_kernels");
  }
  require_equal(
    "positions dtype",
    required_metadata(metadata_, "pygas.input.positions.dtype"),
    "float64");
  require_equal(
    "side_array dtype",
    required_metadata(metadata_, "pygas.input.side_array.dtype"),
    "int64");

  const std::string manifest = read_text(manifest_path_for(model_path));
  require_equal("schema", required_metadata(metadata_, "pygas.schema"), json_string(manifest, "schema"));
  require_equal("route", route_, json_string(manifest, "route"));
  require_equal("torch version", torch_version, json_string(manifest, "torch_version"));
  require_equal("CUDA version", required_metadata(metadata_, "pygas.cuda_version"), json_string(manifest, "cuda_version"));
  if (json_integer(manifest, "abi_version") != metadata_integer(metadata_, "pygas.abi_version") ||
      json_integer(manifest, "n_atoms") != n_atoms ||
      (cv_size > 0 && json_integer(manifest, "cv_size") != cv_size) ||
      json_integer(manifest, "max_edges") != max_edges) {
    throw std::runtime_error("AOTI package and manifest integer ABI fields do not match");
  }
}

torch::Tensor GASModelRunner::input_or_cvonly_default(
  const torch::Dict<std::string, torch::Tensor>& inputs,
  const std::string& name) const
{
  if (inputs.contains(name)) {
    return inputs.at(name);
  }
  if (route_ != "cvonly") {
    throw std::runtime_error("AOTI input is missing: " + name);
  }
  const auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat64);
  if (name == "cv_traj") {
    return torch::empty({0, cv_size_}, options);
  }
  if (name == "bias_traj") {
    return torch::empty({0}, options);
  }
  if (name == "saved_cv_nums") {
    return torch::zeros({}, options.dtype(torch::kInt32));
  }
  throw std::runtime_error("AOTI input is missing: " + name);
}

void GASModelRunner::validate_tensor(
  const std::string& name,
  const torch::Tensor& value) const
{
  const std::string expected_dtype =
    required_metadata(metadata_, "pygas.input." + name + ".dtype");
  const std::string actual_dtype = std::string(c10::toString(value.scalar_type()));
  std::string normalized = actual_dtype;
  for (char& character : normalized) {
    character = static_cast<char>(std::tolower(character));
  }
  if ((expected_dtype == "float64" && value.scalar_type() != torch::kFloat64) ||
      (expected_dtype == "int64" && value.scalar_type() != torch::kInt64) ||
      (expected_dtype == "int32" && value.scalar_type() != torch::kInt32)) {
    throw std::runtime_error(
      "AOTI input " + name + " dtype mismatch: expected " + expected_dtype +
      ", got " + normalized);
  }
  if (!value.is_cuda()) {
    throw std::runtime_error("AOTI input " + name + " must be CUDA");
  }
}

torch::Dict<std::string, torch::Tensor>
GASModelRunner::run(const torch::Dict<std::string, torch::Tensor>& inputs)
{
  if (backend_ == "torchscript") {
    auto raw = torchscript_->forward({inputs}).toGenericDict();
    torch::Dict<std::string, torch::Tensor> outputs;
    for (const auto& item : raw) {
      outputs.insert(item.key().toStringRef(), item.value().toTensor());
    }
    return outputs;
  }

  long saved = -1;
  if (route_ == "bias" || route_ == "cvonly") {
    saved = input_or_cvonly_default(inputs, "saved_cv_nums").item<long>();
    const long declared_history = metadata_integer(metadata_, "pygas.max_cv_nums");
    if (saved < 0 || (max_history_ > 0 && saved > max_history_) ||
        saved > declared_history) {
      throw std::runtime_error("saved_cv_nums is outside the AOTI history capacity");
    }
  }
  std::vector<torch::Tensor> ordered;
  ordered.reserve(input_names_.size());
  for (const std::string& name : input_names_) {
    torch::Tensor value = input_or_cvonly_default(inputs, name);
    if ((name == "cv_traj" || name == "bias_traj") && saved >= 0) {
      if (value.size(0) < saved) {
        throw std::runtime_error(name + " is shorter than saved_cv_nums");
      }
      value = value.narrow(0, 0, saved);
    }
    validate_tensor(name, value);
    ordered.push_back(value.contiguous());
  }
#ifndef GAS_HAS_AOTI
  throw std::runtime_error("Internal error: AOTI execution requested in a TorchScript-only build");
#else
  auto stream = c10::cuda::getCurrentCUDAStream();
  std::vector<torch::Tensor> values = aoti_->run(ordered, stream.stream());
  if (values.size() != output_names_.size()) {
    throw std::runtime_error(
      "AOTI output count mismatch: expected " + std::to_string(output_names_.size()) +
      ", got " + std::to_string(values.size()));
  }
  torch::Dict<std::string, torch::Tensor> outputs;
  for (size_t index = 0; index < values.size(); ++index) {
    outputs.insert(output_names_[index], values[index]);
  }
  return outputs;
#endif
}

torch::Dict<std::string, torch::Tensor>
GASModelRunner::initialize_opes_state(int capacity)
{
  if (route_ != "opes") {
    throw std::runtime_error("initialize_opes_state is only valid for the OPES route");
  }
  if (backend_ == "torchscript") {
    const auto raw = torchscript_->run_method("initialize_state", capacity).toGenericDict();
    torch::Dict<std::string, torch::Tensor> state;
    for (const auto& item : raw) {
      state.insert(item.key().toStringRef(), item.value().toTensor());
    }
    return state;
  }
  if (capacity != metadata_integer(metadata_, "pygas.opes_kernel_capacity")) {
    throw std::runtime_error(
      "AOTI OPES state capacity is static and must equal the package capacity");
  }
  const int cv_size = static_cast<int>(metadata_integer(metadata_, "pygas.cv_size"));
  const auto double_options =
    torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA);
  torch::Dict<std::string, torch::Tensor> state;
  state.insert("opes_kernel_centers", torch::zeros({capacity, cv_size}, double_options));
  state.insert("opes_kernel_sigmas", torch::zeros({capacity, cv_size}, double_options));
  state.insert("opes_kernel_heights", torch::zeros({capacity}, double_options));
  const std::vector<std::string> vector_names = {
    "opes_state_scalars",
    "opes_sigma0",
    "opes_av_cv",
    "opes_av_m2",
    "opes_state_metadata"};
  for (const auto& name : vector_names) {
    const std::string key = "pygas.opes.initial." + name;
    const auto values = parse_number_array(required_metadata(metadata_, key), key);
    state.insert(name, torch::tensor(values, double_options));
  }
  return state;
}
