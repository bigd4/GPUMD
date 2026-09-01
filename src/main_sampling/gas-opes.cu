#include "gas-opes.cuh"
#include "model/read_xyz.cuh"
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace {

constexpr int OPES_STATE_VERSION = 1;
constexpr const char* STATE_KEYS[] = {
  "opes_kernel_centers",
  "opes_kernel_sigmas",
  "opes_kernel_heights",
  "opes_state_scalars",
  "opes_sigma0",
  "opes_av_cv",
  "opes_av_m2",
  "opes_state_metadata"};

std::string trim(std::string value)
{
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return "";
  }
  const auto last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

std::string unquote(std::string value)
{
  value = trim(value);
  if (
    value.size() >= 2 &&
    ((value.front() == '"' && value.back() == '"') ||
     (value.front() == '\'' && value.back() == '\''))) {
    return value.substr(1, value.size() - 2);
  }
  return value;
}

std::unordered_map<std::string, std::string> read_flat_config(const std::string& path)
{
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("Cannot open OPES configuration: " + path);
  }
  std::unordered_map<std::string, std::string> result;
  std::string line;
  while (std::getline(input, line)) {
    const auto hash = line.find('#');
    if (hash != std::string::npos) {
      line = line.substr(0, hash);
    }
    const auto colon = line.find(':');
    if (colon == std::string::npos) {
      continue;
    }
    const std::string key = trim(line.substr(0, colon));
    const std::string value = unquote(line.substr(colon + 1));
    if (!key.empty()) {
      result[key] = value;
    }
  }
  return result;
}

bool parse_bool(const std::string& raw)
{
  std::string value = raw;
  for (char& ch : value) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  if (value == "true" || value == "yes" || value == "on" || value == "1") {
    return true;
  }
  if (value == "false" || value == "no" || value == "off" || value == "0") {
    return false;
  }
  throw std::runtime_error("Expected a boolean value, got: " + raw);
}

void ensure_header(
  const std::string& path,
  const std::vector<std::string>& fields,
  const std::vector<std::pair<std::string, std::string>>& metadata)
{
  std::ostringstream expected;
  expected << "#! FIELDS";
  for (const auto& field : fields) {
    expected << ' ' << field;
  }
  std::ifstream existing(path, std::ios::binary | std::ios::ate);
  if (existing && existing.tellg() > 0) {
    existing.close();
    std::ifstream input(path);
    std::string first_line;
    std::getline(input, first_line);
    if (trim(first_line) != expected.str()) {
      throw std::runtime_error(
        "Existing OPES log has an incompatible FIELDS header: " + path);
    }
    return;
  }
  std::ofstream output(path, std::ios::out);
  if (!output) {
    throw std::runtime_error("Cannot create OPES log: " + path);
  }
  output << expected.str() << '\n';
  for (const auto& item : metadata) {
    output << "#! SET " << item.first << ' ' << item.second << '\n';
  }
}

static __global__ void find_neighbor_list_large_box_opes(
  const float neighbor_rc,
  const int max_neighbors,
  const int n_atoms,
  const int nx,
  const int ny,
  const int nz,
  const Box box,
  const int* __restrict__ cell_count,
  const int* __restrict__ cell_count_sum,
  const int* __restrict__ cell_contents,
  const double* __restrict__ x,
  const double* __restrict__ y,
  const double* __restrict__ z,
  int* nn_radial,
  int* nl_radial)
{
  const int n1 = blockIdx.x * blockDim.x + threadIdx.x;
  if (n1 >= n_atoms) {
    return;
  }
  const double x1 = x[n1];
  const double y1 = y[n1];
  const double z1 = z[n1];
  int count = 0;
  int cell_id, cell_x, cell_y, cell_z;
  find_cell_id(
    box,
    x1,
    y1,
    z1,
    2.0f / neighbor_rc,
    nx,
    ny,
    nz,
    cell_x,
    cell_y,
    cell_z,
    cell_id);
  // The cell edge is neighbor_rc/2, so every axis needs the +/-2 stencil
  // even when that direction is non-periodic.  Periodic boxes with fewer
  // than five cells map multiple offsets to the same cell; suppress those
  // aliases so every directed atom pair is emitted exactly once.
  for (int dz = -2; dz <= 2; ++dz) {
    int neighbor_z = cell_z + dz;
    if (box.pbc_z) {
      neighbor_z = (neighbor_z % nz + nz) % nz;
      bool duplicate_z = false;
      for (int previous = -2; previous < dz; ++previous) {
        const int previous_z = ((cell_z + previous) % nz + nz) % nz;
        duplicate_z = duplicate_z || previous_z == neighbor_z;
      }
      if (duplicate_z)
        continue;
    } else if (neighbor_z < 0 || neighbor_z >= nz) {
      continue;
    }
    for (int dy = -2; dy <= 2; ++dy) {
      int neighbor_y = cell_y + dy;
      if (box.pbc_y) {
        neighbor_y = (neighbor_y % ny + ny) % ny;
        bool duplicate_y = false;
        for (int previous = -2; previous < dy; ++previous) {
          const int previous_y = ((cell_y + previous) % ny + ny) % ny;
          duplicate_y = duplicate_y || previous_y == neighbor_y;
        }
        if (duplicate_y)
          continue;
      } else if (neighbor_y < 0 || neighbor_y >= ny) {
        continue;
      }
      for (int dx = -2; dx <= 2; ++dx) {
        int neighbor_x = cell_x + dx;
        if (box.pbc_x) {
          neighbor_x = (neighbor_x % nx + nx) % nx;
          bool duplicate_x = false;
          for (int previous = -2; previous < dx; ++previous) {
            const int previous_x = ((cell_x + previous) % nx + nx) % nx;
            duplicate_x = duplicate_x || previous_x == neighbor_x;
          }
          if (duplicate_x)
            continue;
        } else if (neighbor_x < 0 || neighbor_x >= nx) {
          continue;
        }
        const int neighbor_cell =
          neighbor_z * nx * ny + neighbor_y * nx + neighbor_x;
        const int atoms_in_cell = cell_count[neighbor_cell];
        const int offset = cell_count_sum[neighbor_cell];
        for (int m = 0; m < atoms_in_cell; ++m) {
          const int n2 = cell_contents[offset + m];
          if (n1 == n2) {
            continue;
          }
          double dx12 = x[n2] - x1;
          double dy12 = y[n2] - y1;
          double dz12 = z[n2] - z1;
          apply_mic(box, dx12, dy12, dz12);
          const double distance2 = dx12 * dx12 + dy12 * dy12 + dz12 * dz12;
          if (distance2 < neighbor_rc * neighbor_rc) {
            if (count < max_neighbors) {
              nl_radial[count * n_atoms + n1] = n2;
            }
            ++count;
          }
        }
      }
    }
  }
  nn_radial[n1] = count;
}

torch::Tensor neighbor_indices(torch::Tensor& nl, int n_atoms)
{
  const int max_neighbors = static_cast<int>(nl.size(0)) / n_atoms;
  auto starts = torch::arange(0, n_atoms, torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA));
  starts = starts.repeat({max_neighbors});
  const auto mask = nl >= 0;
  return torch::stack({starts.masked_select(mask), nl.masked_select(mask).to(torch::kLong)}, 1);
}

torch::jit::script::Module load_model(const std::string& path)
{
  torch::jit::setGraphExecutorOptimize(true);
  auto model = torch::jit::load(path, torch::kCUDA);
  model.eval();
  return model;
}

} // namespace

OPESConfig OPESConfig::from_file(const std::string& path, int n_atoms)
{
  OPESConfig result;
  Config base;
  base.n_atoms = n_atoms;
  result.common = Config::fromFile(path, base);
  const auto values = read_flat_config(path);
  if (values.count("opes_max_kernels"))
    result.max_kernels = std::stoi(values.at("opes_max_kernels"));
  if (values.count("opes_state_interval"))
    result.state_interval = std::stoi(values.at("opes_state_interval"));
  if (values.count("opes_state_path"))
    result.state_path = values.at("opes_state_path");
  if (values.count("opes_kernels_path"))
    result.kernels_path = values.at("opes_kernels_path");
  if (values.count("opes_stop_on_status"))
    result.stop_on_status = parse_bool(values.at("opes_stop_on_status"));
  if (values.count("opes_stop_status"))
    result.stop_status = std::stoi(values.at("opes_stop_status"));

  if (result.common.n_atoms != n_atoms)
    throw std::runtime_error("GAScfg n_atoms does not match model.xyz for OPES.");
  if (result.common.cv_size <= 0 || result.common.cv_log_interval <= 0)
    throw std::runtime_error("OPES cv_size and cv_log_interval must be positive.");
  if (result.common.neighbor_rc <= 0 || result.common.max_neighbors <= 0)
    throw std::runtime_error("OPES neighbor_rc and max_neighbors must be positive.");
  if (result.max_kernels <= 0 || result.state_interval < 0)
    throw std::runtime_error("opes_max_kernels must be positive and opes_state_interval non-negative.");
  if (result.common.cv_log_path.empty() || result.state_path.empty() || result.kernels_path.empty())
    throw std::runtime_error("OPES log, state, and kernel paths must not be empty.");
  return result;
}

void OPESConfig::print() const
{
  common.print();
  std::cout << "[OPEScfg] Max kernels: " << max_kernels << '\n';
  std::cout << "[OPEScfg] State path: " << state_path << '\n';
  std::cout << "[OPEScfg] State interval: " << state_interval << '\n';
  std::cout << "[OPEScfg] Kernels path: " << kernels_path << '\n';
  std::cout << "[OPEScfg] Stop on status: " << stop_on_status << '\n';
  std::cout << "[OPEScfg] Stop status: " << stop_status << std::endl;
}

TorchOPES::TorchOPES(const std::string& model_path, const std::string& cfg_path, int n_atoms)
  : n_atoms_(n_atoms)
{
  initialize(model_path, cfg_path);
  initialize_state();
  initialized_ = true;
}

TorchOPES::TorchOPES(
  const std::string& model_path,
  const std::string& cfg_path,
  const std::string& state_path,
  int n_atoms)
  : n_atoms_(n_atoms)
{
  initialize(model_path, cfg_path);
  config_.state_path = state_path;
  load_state(state_path);
  initialized_ = true;
}

TorchOPES::~TorchOPES()
{
  if (initialized_ && state_dirty_) {
    try {
      checkpoint();
    } catch (const std::exception& error) {
      std::cerr << "[OPES-Warn] Final checkpoint failed: " << error.what() << std::endl;
    }
  }
}

void TorchOPES::initialize(const std::string& model_path, const std::string& cfg_path)
{
  model_ = load_model(model_path);
  config_ = OPESConfig::from_file(cfg_path, n_atoms_);
  config_.print();
  cell_count_.resize(n_atoms_);
  cell_count_sum_.resize(n_atoms_);
  cell_contents_.resize(n_atoms_);
  nn_radial_.resize(n_atoms_);
  nl_radial_.resize(n_atoms_ * config_.common.max_neighbors);
  cpu_box_.resize(9);
  std::cout << "[OPES-Info] Loaded TorchScript model " << model_path << std::endl;
}

void TorchOPES::initialize_state()
{
  const auto value = model_.run_method("initialize_state", config_.max_kernels);
  const auto dictionary = value.toGenericDict();
  for (const char* key : STATE_KEYS) {
    auto found = dictionary.find(c10::IValue(std::string(key)));
    if (found == dictionary.end()) {
      throw std::runtime_error(std::string("OPES initialize_state is missing key: ") + key);
    }
    state_[key] = found->value().toTensor();
  }
  const auto scalars = state_.at("opes_state_scalars");
  if (scalars.numel() < 14 || std::lround(scalars[13].item<double>()) != OPES_STATE_VERSION) {
    throw std::runtime_error("Unsupported OPES state version returned by TorchScript model.");
  }
  now_step_ = 0;
  status_ = 0;
  state_dirty_ = true;
}

void TorchOPES::load_state(const std::string& path)
{
  torch::serialize::InputArchive archive;
  archive.load_from(path);
  for (const char* key : STATE_KEYS) {
    torch::Tensor tensor;
    archive.read(key, tensor);
    state_[key] = tensor.to(torch::kCUDA);
  }
  torch::Tensor step;
  archive.read("next_step", step);
  now_step_ = step.item<int64_t>();
  const auto scalars = state_.at("opes_state_scalars");
  if (scalars.numel() < 14 || std::lround(scalars[13].item<double>()) != OPES_STATE_VERSION) {
    throw std::runtime_error("Unsupported OPES checkpoint state version: " + path);
  }
  const int64_t saved_capacity = state_.at("opes_kernel_centers").size(0);
  if (saved_capacity > config_.max_kernels) {
    throw std::runtime_error(
      "OPES checkpoint capacity exceeds opes_max_kernels; increase the configured capacity.");
  }
  if (saved_capacity < config_.max_kernels) {
    const int64_t extra = config_.max_kernels - saved_capacity;
    const auto center_options = state_.at("opes_kernel_centers").options();
    const auto height_options = state_.at("opes_kernel_heights").options();
    state_["opes_kernel_centers"] = torch::cat(
      {state_.at("opes_kernel_centers"),
       torch::zeros({extra, config_.common.cv_size}, center_options)},
      0);
    state_["opes_kernel_sigmas"] = torch::cat(
      {state_.at("opes_kernel_sigmas"),
       torch::zeros({extra, config_.common.cv_size}, center_options)},
      0);
    state_["opes_kernel_heights"] = torch::cat(
      {state_.at("opes_kernel_heights"), torch::zeros({extra}, height_options)}, 0);
    std::cout << "[OPES-Info] Expanded checkpoint kernel capacity from " << saved_capacity
              << " to " << config_.max_kernels << std::endl;
  }
  const auto expected_state =
    model_.run_method("initialize_state", config_.max_kernels).toGenericDict();
  const auto expected_entry =
    expected_state.find(c10::IValue(std::string("opes_state_metadata")));
  if (expected_entry == expected_state.end()) {
    throw std::runtime_error("OPES model does not expose checkpoint compatibility metadata.");
  }
  const auto actual_metadata = state_.at("opes_state_metadata").reshape({-1});
  const auto expected_metadata = expected_entry->value().toTensor().reshape({-1});
  if (
    actual_metadata.sizes() != expected_metadata.sizes() ||
    !torch::allclose(actual_metadata, expected_metadata, 1.0e-12, 1.0e-14)) {
    throw std::runtime_error(
      "OPES checkpoint scientific/algorithmic settings do not match the loaded model.");
  }
  state_["opes_state_scalars"][12] = 0.0;
  // GPUMD evaluates forces once before entering the integration loop.  For a
  // loaded paired checkpoint, that position has already advanced OPES state
  // in the preceding run.  Recompute its force, but discard the model's state
  // update and do not advance/log the OPES step a second time.
  restart_force_refresh_pending_ = true;
  std::cout << "[OPES-Info] Exact state restored from " << path << " at next step " << now_step_
            << std::endl;
  state_dirty_ = false;
}

void TorchOPES::checkpoint()
{
  const std::string temporary = config_.state_path + ".tmp";
  torch::serialize::OutputArchive archive;
  for (const char* key : STATE_KEYS) {
    archive.write(key, state_.at(key).detach().to(torch::kCPU));
  }
  archive.write("next_step", torch::tensor(static_cast<int64_t>(now_step_), torch::kInt64));
  archive.save_to(temporary);
  std::error_code error;
  std::filesystem::rename(temporary, config_.state_path, error);
  if (error) {
    std::filesystem::remove(config_.state_path, error);
    error.clear();
    std::filesystem::rename(temporary, config_.state_path, error);
  }
  if (error) {
    throw std::runtime_error("Cannot atomically install OPES checkpoint: " + error.message());
  }
  state_dirty_ = false;
}

void TorchOPES::update_state(const c10::impl::GenericDict& output)
{
  for (const char* key : STATE_KEYS) {
    auto found = output.find(c10::IValue(std::string(key)));
    if (found == output.end()) {
      throw std::runtime_error(std::string("OPES forward output is missing state key: ") + key);
    }
    state_[key] = found->value().toTensor();
  }
  state_dirty_ = true;
}

void TorchOPES::compute_large_box(Box& box, const GPU_Vector<double>& positions)
{
  constexpr int block_size = 64;
  const int grid_size = (n_atoms_ - 1) / block_size + 1;
  const double cell_cutoff = 0.5 * config_.common.neighbor_rc;
  int bins[3];
  box.get_num_bins(cell_cutoff, bins);
  find_cell_list(
    cell_cutoff,
    bins,
    box,
    positions,
    cell_count_,
    cell_count_sum_,
    cell_contents_);
  find_neighbor_list_large_box_opes<<<grid_size, block_size>>>(
    config_.common.neighbor_rc,
    config_.common.max_neighbors,
    n_atoms_,
    bins[0],
    bins[1],
    bins[2],
    box,
    cell_count_.data(),
    cell_count_sum_.data(),
    cell_contents_.data(),
    positions.data(),
    positions.data() + n_atoms_,
    positions.data() + 2 * n_atoms_,
    nn_radial_.data(),
    nl_radial_.data());
  torch::cuda::synchronize();
}

void TorchOPES::get_neighbor_list(Box& box, const GPU_Vector<double>& positions)
{
  nn_radial_.fill(-1);
  nl_radial_.fill(-1);
  compute_large_box(box, positions);
}

torch::Tensor TorchOPES::from_cuda_memory(double* array, int size)
{
  return torch::from_blob(
    array, {size}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA));
}

void TorchOPES::box_to_tri(Box& box)
{
  cpu_box_[0] = box.cpu_h[0];
  cpu_box_[1] = box.cpu_h[3];
  cpu_box_[2] = box.cpu_h[6];
  cpu_box_[3] = box.cpu_h[1];
  cpu_box_[4] = box.cpu_h[4];
  cpu_box_[5] = box.cpu_h[7];
  cpu_box_[6] = box.cpu_h[2];
  cpu_box_[7] = box.cpu_h[5];
  cpu_box_[8] = box.cpu_h[8];
}

void TorchOPES::log_colvar(const c10::impl::GenericDict& output, int evaluated_step)
{
  std::vector<std::string> fields = {"step", "bias"};
  for (int dim = 0; dim < config_.common.cv_size; ++dim)
    fields.push_back("cv_" + std::to_string(dim));
  const std::vector<std::string> tail = {
    "mean_bias_force",
    "extra_bias",
    "rct",
    "zed",
    "neff",
    "nker",
    "work",
    "excluded",
    "status",
    "deposited",
    "merge_count",
    "logweight",
    "probability"};
  fields.insert(fields.end(), tail.begin(), tail.end());
  ensure_header(
    config_.common.cv_log_path,
    fields,
    {{"opes_state_version", std::to_string(OPES_STATE_VERSION)},
     {"cv_size", std::to_string(config_.common.cv_size)}});

  const auto cv = output.at("cv_now").toTensor().detach().to(torch::kCPU).reshape({-1});
  const auto force = output.at("forces").toTensor().detach();
  const double mean_force = force.norm(2, 1).mean().item<double>();
  std::ofstream file(config_.common.cv_log_path, std::ios::app);
  file << std::setprecision(17) << evaluated_step << ' '
       << output.at("bias").toTensor().item<double>();
  for (int dim = 0; dim < config_.common.cv_size; ++dim)
    file << ' ' << cv[dim].item<double>();
  file << ' ' << mean_force << ' '
       << output.at("opes_extra_bias").toTensor().item<double>() << ' '
       << output.at("opes_rct").toTensor().item<double>() << ' '
       << output.at("opes_zed").toTensor().item<double>() << ' '
       << output.at("opes_neff").toTensor().item<double>() << ' '
       << output.at("opes_nker").toTensor().item<double>() << ' '
       << output.at("opes_work").toTensor().item<double>() << ' '
       << output.at("opes_excluded").toTensor().item<double>() << ' ' << status_ << ' '
       << output.at("opes_deposited").toTensor().item<double>() << ' '
       << output.at("opes_merge_count").toTensor().item<double>() << ' '
       << output.at("opes_logweight").toTensor().item<double>() << ' '
       << output.at("opes_probability").toTensor().item<double>() << '\n';
}

void TorchOPES::log_kernel(const c10::impl::GenericDict& output, int evaluated_step)
{
  if (output.at("opes_deposited").toTensor().item<double>() < 0.5) {
    return;
  }
  std::vector<std::string> fields = {"step"};
  for (int dim = 0; dim < config_.common.cv_size; ++dim)
    fields.push_back("cv_" + std::to_string(dim));
  for (int dim = 0; dim < config_.common.cv_size; ++dim)
    fields.push_back("sigma_cv_" + std::to_string(dim));
  const std::vector<std::string> tail = {"height", "logweight", "merge_count", "nker"};
  fields.insert(fields.end(), tail.begin(), tail.end());
  ensure_header(
    config_.kernels_path,
    fields,
    {{"opes_state_version", std::to_string(OPES_STATE_VERSION)}});
  const auto cv = output.at("cv_now").toTensor().detach().to(torch::kCPU).reshape({-1});
  const auto sigma = output.at("opes_last_sigma").toTensor().detach().to(torch::kCPU).reshape({-1});
  std::ofstream file(config_.kernels_path, std::ios::app);
  file << std::setprecision(17) << evaluated_step;
  for (int dim = 0; dim < config_.common.cv_size; ++dim)
    file << ' ' << cv[dim].item<double>();
  for (int dim = 0; dim < config_.common.cv_size; ++dim)
    file << ' ' << sigma[dim].item<double>();
  file << ' ' << output.at("opes_last_height").toTensor().item<double>() << ' '
       << output.at("opes_logweight").toTensor().item<double>() << ' '
       << output.at("opes_merge_count").toTensor().item<double>() << ' '
       << output.at("opes_nker").toTensor().item<double>() << '\n';
}

void TorchOPES::compute(
  Box& box,
  const GPU_Vector<int>&,
  const GPU_Vector<double>& positions,
  GPU_Vector<double>& potentials,
  GPU_Vector<double>& forces,
  GPU_Vector<double>& virial)
{
  if (positions.size() / 3 != n_atoms_)
    throw std::runtime_error("n_atoms in OPES GAScfg differs from model.xyz.");
  box_to_tri(box);
  get_neighbor_list(box, positions);

  auto nn = torch::from_blob(
    nn_radial_.data(),
    {n_atoms_},
    torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA));
  if (nn.max().item<int>() > config_.common.max_neighbors) {
    throw std::runtime_error("OPES neighbor list exceeded max_neighbors; increase GAScfg capacity.");
  }
  auto nl = torch::from_blob(
    nl_radial_.data(),
    {n_atoms_ * config_.common.max_neighbors},
    torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA));
  auto side_array = neighbor_indices(nl, n_atoms_);
  auto position = from_cuda_memory((double*)positions.data(), positions.size())
                    .detach()
                    .clone()
                    .reshape({3, -1})
                    .transpose(0, 1);
  auto force = from_cuda_memory(forces.data(), positions.size()).reshape({3, -1}).transpose(0, 1);
  auto potential = from_cuda_memory(potentials.data(), n_atoms_);
  auto atom_virial = from_cuda_memory(virial.data(), positions.size() * 3)
                        .reshape({9, -1})
                        .transpose(0, 1);
  auto cell = torch::from_blob(cpu_box_.data(), {9}, torch::kFloat64)
                .to(torch::kCUDA)
                .reshape({3, 3});

  torch::Dict<std::string, torch::Tensor> inputs;
  inputs.insert("positions", position);
  inputs.insert("cell", cell);
  inputs.insert("side_array", side_array);
  inputs.insert(
    "opes_step",
    torch::tensor(static_cast<int64_t>(now_step_), torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA)));
  for (const char* key : STATE_KEYS)
    inputs.insert(key, state_.at(key));

  const int evaluated_step = now_step_;
  auto output = model_.forward({inputs}).toGenericDict();
  status_ = output.at("status").toTensor().item<int64_t>();
  if (!restart_force_refresh_pending_)
    update_state(output);
  const auto bias = output.at("bias").toTensor();
  potential += bias.unsqueeze(0) / n_atoms_;
  force += output.at("forces").toTensor();
  atom_virial += output.at("virial").toTensor().unsqueeze(0) / n_atoms_;
  torch::cuda::synchronize();

  if (restart_force_refresh_pending_) {
    restart_force_refresh_pending_ = false;
    return;
  }

  if (evaluated_step % config_.common.cv_log_interval == 0)
    log_colvar(output, evaluated_step);
  log_kernel(output, evaluated_step);

  const bool capacity_exceeded =
    state_.at("opes_state_scalars")[12].item<double>() > 0.5;
  if (capacity_exceeded) {
    // The model rolled back the attempted update.  Do not overwrite the last
    // paired OPES/atomic checkpoint: this failure is detected after the MD
    // engine's first half-step, so the current in-memory atomic state is not a
    // valid full-step restart.  The previous periodic checkpoint remains the
    // scientifically safe recovery point.
    state_dirty_ = false;
    throw std::runtime_error(
      "OPES kernel capacity exceeded and the attempted update was rolled back. "
      "The last paired checkpoint was preserved; increase opes_max_kernels and restart "
      "from the matching OPES_STATE.pt plus atomic restart.xyz.");
  }
  ++now_step_;
  if (
    config_.state_interval > 0 && evaluated_step > 0 &&
    evaluated_step % config_.state_interval == 0) {
    checkpoint();
  }
}

bool TorchOPES::should_stop() const
{
  if (!config_.stop_on_status)
    return false;
  return config_.stop_status == 0 ? status_ != 0 : status_ == config_.stop_status;
}

std::unique_ptr<TorchOPES>
TorchOPES::parse_OPES(const char** param, int num_param, int number_of_atoms)
{
  if (num_param == 3) {
    return std::make_unique<TorchOPES>(param[1], param[2], number_of_atoms);
  }
  if (num_param == 4) {
    return std::make_unique<TorchOPES>(param[1], param[2], param[3], number_of_atoms);
  }
  throw std::runtime_error(
    "OPES syntax: OPES model.pt GAScfg.yaml [OPES_STATE.pt]");
}
