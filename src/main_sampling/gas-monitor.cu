// #ifdef USE_GAS
#include "gas-monitor.cuh"
#include "model/read_xyz.cuh"

namespace{
//   std::vector<std::vector<double>> readFileToVector(const std::string& filename) {
//     std::ifstream file(filename);  // 打开文件
//     if (!file.is_open()) {
//         std::cerr << "无法打开文件: " << filename << std::endl;
//         throw std::runtime_error("无法打开文件");
//     }

//     std::vector<std::vector<double>> data;  // 存储每一行的浮点数
//     std::string line;

//     // 逐行读取文件
//     while (std::getline(file, line)) {
//         std::stringstream ss(line);
//         std::vector<double> row;
//         double value;

//         // 逐个浮点数读取并存入当前行的vector
//         while (ss >> value) {
//             row.push_back(value);
//         }

//         data.push_back(row);  // 将当前行添加到data中
//     }

//     file.close();  // 关闭文件
//     return data;
// }


// torch::Tensor vectorToTensor(const std::vector<std::vector<double>>& data) {
//     // 获取数据的维度
//     auto m = data.size();           // 行数
//     auto n = data.empty() ? 0 : data[0].size();  // 列数

//     // 将二维std::vector平展为一维std::vector
//     std::vector<double> flat_data;
//     for (const auto& row : data) {
//         flat_data.insert(flat_data.end(), row.begin(), row.end());
//     }

//     // 创建张量 (m, n)，数据类型为 double
//     torch::Tensor tensor = torch::from_blob(flat_data.data(), {(long)m, (long)n}, torch::kFloat64);

//     return tensor.clone();  // 返回一个副本，确保数据不受外部修改的影响
// }

torch::Tensor NL2Indices(torch::Tensor& torch_NL,int n_atoms_NL){
    int NL_size = torch_NL.size(0);
    int max_neighbor = NL_size/n_atoms_NL;
    torch::Tensor temp = torch::arange(0, n_atoms_NL,torch::kCUDA);
    torch::Tensor start_array = temp.repeat({max_neighbor});
    torch::Tensor mask = torch_NL >= 0;

    start_array = start_array.masked_select(mask);
    torch::Tensor end_array = torch_NL.masked_select(mask);
    torch::Tensor side_array = torch::stack({start_array, end_array}, 1);
    return side_array;
}


}



static __global__ void find_neighbor_list_large_box_gas(
  const float neighbor_rc,
  const int N,
  const int N1,
  const int N2,
  const int nx,
  const int ny,
  const int nz,
  const Box box,
  // const int* g_type,
  const int* __restrict__ g_cell_count,
  const int* __restrict__ g_cell_count_sum,
  const int* __restrict__ g_cell_contents,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  int* g_NN_radial,
  int* g_NL_radial)
{
  int n1 = blockIdx.x * blockDim.x + threadIdx.x + N1;
  if (n1 >= N2) {
    return;
  }

  double x1 = g_x[n1];
  double y1 = g_y[n1];
  double z1 = g_z[n1];
  int count_radial = 0;

  int cell_id;
  int cell_id_x;
  int cell_id_y;
  int cell_id_z;
  find_cell_id(
    box,
    x1,
    y1,
    z1,
    2.0f / neighbor_rc,
    nx,
    ny,
    nz,
    cell_id_x,
    cell_id_y,
    cell_id_z,
    cell_id);

  const int z_lim = box.pbc_z ? 2 : 0;
  const int y_lim = box.pbc_y ? 2 : 0;
  const int x_lim = box.pbc_x ? 2 : 0;

  for (int zz = -z_lim; zz <= z_lim; ++zz) {
    for (int yy = -y_lim; yy <= y_lim; ++yy) {
      for (int xx = -x_lim; xx <= x_lim; ++xx) {
        int neighbor_cell = cell_id + zz * nx * ny + yy * nx + xx;
        if (cell_id_x + xx < 0)
          neighbor_cell += nx;
        if (cell_id_x + xx >= nx)
          neighbor_cell -= nx;
        if (cell_id_y + yy < 0)
          neighbor_cell += ny * nx;
        if (cell_id_y + yy >= ny)
          neighbor_cell -= ny * nx;
        if (cell_id_z + zz < 0)
          neighbor_cell += nz * ny * nx;
        if (cell_id_z + zz >= nz)
          neighbor_cell -= nz * ny * nx;

        const int num_atoms_neighbor_cell = g_cell_count[neighbor_cell];
        const int num_atoms_previous_cells = g_cell_count_sum[neighbor_cell];

        for (int m = 0; m < num_atoms_neighbor_cell; ++m) {
          const int n2 = g_cell_contents[num_atoms_previous_cells + m];

          if (n2 < N1 || n2 >= N2 || n1 == n2) {
            continue;
          }

          double x12double = g_x[n2] - x1;
          double y12double = g_y[n2] - y1;
          double z12double = g_z[n2] - z1;
          apply_mic(box, x12double, y12double, z12double);
          float x12 = float(x12double), y12 = float(y12double), z12 = float(z12double);
          float d12_square = x12 * x12 + y12 * y12 + z12 * z12;
          
          float rc_radial = neighbor_rc;

          if (d12_square >= rc_radial * rc_radial) {
            continue;
          }
          g_NL_radial[count_radial++ * N + n1] = n2;
        }
      }
    }
  }

  g_NN_radial[n1] = count_radial;
}

// torch::Tensor NL2Indices(torch::Tensor& torch_NL,int n_atoms_NL){
//     int NL_size = torch_NL.size(0);
//     int max_neighbor = NL_size/n_atoms_NL;
//     torch::Tensor temp = torch::arange(0, n_atoms_NL,torch::kCUDA);
//     torch::Tensor start_array = temp.repeat({max_neighbor});
//     torch::Tensor mask = torch_NL >= 0;

//     start_array = start_array.masked_select(mask);
//     torch::Tensor end_array = torch_NL.masked_select(mask);
//     torch::Tensor side_array = torch::stack({start_array, end_array}, 1);
//     return side_array;
// }


void TorchMonitor::compute_large_box(
  Box& box,
  const GPU_Vector<double>& position_per_atom)
{
  const int BLOCK_SIZE = 64;
  const int N = n_atoms_;
  //N1~N2是要计算的原子的序号
//   const int grid_size = (this->N2_ - this->N1_ - 1) / BLOCK_SIZE + 1;
  const int grid_size = (N - 1) / BLOCK_SIZE + 1;

  const double rc_cell_list = 0.5 * config.neighbor_rc;

  int num_bins[3];
  box.get_num_bins(rc_cell_list, num_bins);

  find_cell_list(
    rc_cell_list,
    num_bins,
    box,
    position_per_atom,
    this->cell_count,
    this->cell_count_sum,
    this->cell_contents);

  find_neighbor_list_large_box_gas<<<grid_size, BLOCK_SIZE>>>(
    config.neighbor_rc,
    N,
    0,// this->N1_,
    N,// this->N2_,
    num_bins[0],
    num_bins[1],
    num_bins[2],
    box,
    this->cell_count.data(),
    this->cell_count_sum.data(),
    this->cell_contents.data(),
    position_per_atom.data(),
    position_per_atom.data() + N,
    position_per_atom.data() + N * 2,
    this->NN_radial.data(),
    this->NL_radial.data());
    torch::cuda::synchronize();
}

void TorchMonitor::get_neighbor_list(Box& box,const GPU_Vector<double>& position_per_atom){
  this->NN_radial.fill(-1);
  this->NL_radial.fill(-1);
  this->compute_large_box(box,position_per_atom);
}
TorchMonitor::TorchMonitor(int n_atoms):TorchMonitor("GASCVModel.pt","GAScfg.yaml",n_atoms){}

TorchMonitor::TorchMonitor(std::string model_path,std::string cfg_path,int n_atoms){
    this->n_atoms_ = n_atoms;
    // 读取文件，设定参数
    try {
        // torch::jit::GraphOptimizerEnabledGuard guard{true};
        torch::jit::setGraphExecutorOptimize(true);
        // 加载 TorchScript 模型
        model = torch::jit::load(model_path, torch::kCUDA);
        model.eval(); // 设置为评估模式
        std::cout << "[GAS-Info] GASCVModel loaded successfully from " << model_path << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model: "<< model_path << e.what() << std::endl;
        throw e;
    }
    // 接受 GASConfig 参数
    try {
        config = PSConfig::fromFile(cfg_path);
        std::cout << "[GAS-Info] GASConfig loaded successfully from " << cfg_path << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the Config: "<< cfg_path << e.what() << std::endl;
        throw e;
    }

    cell_count.resize(n_atoms_);
    cell_count_sum.resize(n_atoms_);
    cell_contents.resize(n_atoms_);
    NN_radial.resize(n_atoms_);
    NL_radial.resize(n_atoms_*config.max_neighbors);

    torch_now_cvs = torch::empty({config.cv_size},torch::dtype(torch::kFloat64).device(torch::kCUDA));
    torch_bias = torch::empty({},  torch::dtype(torch::kFloat64).device(torch::kCUDA));

    cpu_b_vector = std::vector<double>(9); // Box
    target_stage = config.target_stage;
    torch::cuda::synchronize();

}

TorchMonitor::TorchMonitor(std::string model_path,std::string cfg_path,std::string gaussian_path,int n_atoms){
    this->n_atoms_ = n_atoms;
    // 读取文件，设定参数
    try {
        // torch::jit::GraphOptimizerEnabledGuard guard{true};
        torch::jit::setGraphExecutorOptimize(true);
        // 加载 TorchScript 模型
        model = torch::jit::load(model_path, torch::kCUDA);
        model.eval(); // 设置为评估模式
        std::cout << "[GAS-Info] GASCVModel loaded successfully from " << model_path << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model: "<< model_path << e.what() << std::endl;
        throw e;
    }
    // 接受 GASConfig 参数
    try {
        config = PSConfig::fromFile(cfg_path);
        std::cout << "[GAS-Info] GASConfig loaded successfully from " << cfg_path << std::endl;
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the Config: "<< cfg_path << e.what() << std::endl;
        throw e;
    }

    cell_count.resize(n_atoms_);
    cell_count_sum.resize(n_atoms_);
    cell_contents.resize(n_atoms_);
    NN_radial.resize(n_atoms_);
    NL_radial.resize(n_atoms_*config.max_neighbors);
    
    torch_now_cvs = torch::empty({config.cv_size},  torch::dtype(torch::kFloat64).device(torch::kCUDA));
    torch_bias = torch::empty({},  torch::dtype(torch::kFloat64).device(torch::kCUDA));

    debug_interval = config.debug_interval;

    cpu_b_vector = std::vector<double>(9); // Box
    torch::cuda::synchronize();

}


torch::Dict<std::string, torch::Tensor> TorchMonitor::predict(
    const torch::Dict<std::string, torch::Tensor>& inputs) {
    // try {
        // 将输入传递给模型
        auto result = model.forward({inputs}).toGenericDict();
        // 要花括号吗？
        // 转换返回值为 torch::Dict
        torch::Dict<std::string, torch::Tensor> outputs;
        for (const auto& item : result) {
            auto key = item.key().toStringRef();
            auto value = item.value().toTensor();
            // #ifdef USE_GAS_DEBUG
            if (config.debug_interval!=0){
              if(key.compare("side_array") && key.compare("cv_traj") && key.compare("") && now_step%debug_interval==0){std::cout<<key<<value<<std::endl;}
            }
            // #endif
            outputs.insert(key, value);
        }
        return outputs;
    // } catch (const c10::Error& e) {
    //     std::cerr << "推理失败: " << e.what() << std::endl;
    //     throw;
    // }
}

torch::Tensor TorchMonitor::select_cv_output(const torch::Dict<std::string, torch::Tensor>& outputs) {
    torch::Tensor cvs;
    if (outputs.contains("cv_now")) {
        cvs = outputs.at("cv_now");
    } else if (outputs.contains("cv")) {
        cvs = outputs.at("cv");
    } else if (outputs.contains("commitor")) {
        cvs = outputs.at("commitor");
    } else {
        throw std::runtime_error("GAS monitor model output must contain cv_now, cv, or commitor.");
    }
    cvs = cvs.reshape({-1}).to(torch::kFloat64);
    if (!cvs.is_cuda()) {
        cvs = cvs.to(torch::kCUDA);
    }
    return cvs;
}

int TorchMonitor::evaluate_ffs_event(const torch::Tensor& cvs) {
    torch::Tensor cpu_cvs = cvs.reshape({-1}).to(torch::kCPU).to(torch::kFloat64);
    const int cv_count = static_cast<int>(cpu_cvs.size(0));
    if (config.ffs_cv_index < 0 || config.ffs_cv_index >= cv_count) {
        throw std::runtime_error("ffs_cv_index is out of bounds for GAS monitor CV output.");
    }
    const double cv = cpu_cvs[config.ffs_cv_index].item<double>();
    if (config.ffs_direction >= 0) {
        if (cv >= config.ffs_target_interface) {
            return 1;
        }
        if (config.ffs_stop_on_fail && cv <= config.ffs_fail_interface) {
            return -1;
        }
    } else {
        if (cv <= config.ffs_target_interface) {
            return 1;
        }
        if (config.ffs_stop_on_fail && cv >= config.ffs_fail_interface) {
            return -1;
        }
    }
    return 0;
}

bool TorchMonitor::should_stop_from_outputs(
    const torch::Dict<std::string, torch::Tensor>& outputs,
    int legacy_target_stage) {
    torch_now_cvs = select_cv_output(outputs);

    int event_status = 0;
    if (config.ffs_enabled) {
        event_status = evaluate_ffs_event(torch_now_cvs);
    } else if (outputs.contains("status")) {
        const int status = outputs.at("status").to(torch::kCPU).item<int>();
        if (legacy_target_stage <= 1) {
            event_status = (status == 0) ? 0 : 1;
        } else if (status >= legacy_target_stage) {
            event_status = 1;
        } else if (status == 0) {
            event_status = -1;
        }
    } else {
        throw std::runtime_error("GAS monitor needs either a model status output or ffs_enabled YAML fields.");
    }

    torch_bias = torch::tensor(
        {static_cast<double>(event_status)},
        torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA));
    return event_status != 0;
}

bool TorchMonitor::process(
    Box& box,
    const GPU_Vector<double>& positions,
    int target_stage){
    int dynamic_vector_size = positions.size();
    int n_atoms = dynamic_vector_size/3;    
    this->box_to_tri(box);
    this->get_neighbor_list(box,positions);
    torch::cuda::synchronize();

    torch::Tensor torch_pos = _FromCudaMemory((double*)positions.data(),dynamic_vector_size).detach().clone().reshape({3,-1}).transpose(0,1);
    torch::Tensor torch_cell = torch::from_blob(cpu_b_vector.data(), {9}, torch::dtype(torch::kFloat64)).to(torch::kCUDA).reshape({3,3});

    torch_NL = torch::from_blob(NL_radial.data(), {n_atoms*config.max_neighbors}, torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA)).reshape(-1);
    torch::cuda::synchronize();
    torch::Tensor side_array = NL2Indices(torch_NL,n_atoms);


    torch::Dict<std::string, torch::Tensor> inputs;
    inputs.insert("positions", torch_pos);
    inputs.insert("cell", torch_cell);
    inputs.insert("side_array",side_array);;
    torch::cuda::synchronize();
    //计算和取出输出
    auto output_dict = this->predict(inputs);
    torch::cuda::synchronize();
    const bool should_stop = this->should_stop_from_outputs(output_dict, target_stage);
    torch::cuda::synchronize();
    if(now_step%config.cv_log_interval==0){
      this->logCV_runtime();
    }
    now_step++;

    if(should_stop){
      this->logCV_runtime();
      return true;
    }
    return false;
    }

bool TorchMonitor::process(
    Box& box,
    const GPU_Vector<double>& positions){
    int dynamic_vector_size = positions.size();
    int n_atoms = dynamic_vector_size/3;    
    this->box_to_tri(box);
    this->get_neighbor_list(box,positions);
    torch::cuda::synchronize();

    torch::Tensor torch_pos = _FromCudaMemory((double*)positions.data(),dynamic_vector_size).detach().clone().reshape({3,-1}).transpose(0,1);
    torch::Tensor torch_cell = torch::from_blob(cpu_b_vector.data(), {9}, torch::dtype(torch::kFloat64)).to(torch::kCUDA).reshape({3,3});

    torch_NL = torch::from_blob(NL_radial.data(), {n_atoms*config.max_neighbors}, torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA)).reshape(-1);
    torch::cuda::synchronize();
    torch::Tensor side_array = NL2Indices(torch_NL,n_atoms);


    torch::Dict<std::string, torch::Tensor> inputs;
    inputs.insert("positions", torch_pos);
    inputs.insert("cell", torch_cell);
    inputs.insert("side_array",side_array);;
    torch::cuda::synchronize();
    //计算和取出输出
    auto output_dict = this->predict(inputs);
    torch::cuda::synchronize();
    const bool should_stop = this->should_stop_from_outputs(output_dict, target_stage);
    torch::cuda::synchronize();
    if(now_step%config.cv_log_interval==0){
      this->logCV_runtime();
    }
    now_step++;

    if(should_stop){
      this->logCV_runtime();
      return true;
    }
    return false;
    }


// 将cuda指针用torch视图读取的代码
torch::Tensor TorchMonitor::_FromCudaMemory(double* d_array, int size) {
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA);
    return torch::from_blob(d_array, {size}, options);
}


void TorchMonitor::logCV_runtime(void){
    auto log_cv_tensor = torch_now_cvs;
    torch::Tensor cv_cpu_tensor = log_cv_tensor.to(torch::kCPU);
    torch::Tensor bias_cpu_tensor = this->torch_bias.to(torch::kCPU);
    torch::Tensor cpu_tensor = torch::cat({bias_cpu_tensor.squeeze(0).reshape({-1}),cv_cpu_tensor},0).reshape({-1});

    // 创建一个 TensorAccessor 对象来访问张量数据
    auto tensor_data = cpu_tensor.accessor<double, 1>();
    // 打开文件
    std::string log_name = "GASCVlog.txt";

    std::ofstream file(log_name,std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " <<log_name<< std::endl;
        return;
    }
    // 写入张量数据
    for (int j = 0; j < cpu_tensor.size(0); ++j) {
                file << tensor_data[j];
                if (j < cpu_tensor.size(0) - 1) {
                    file << "\t\t";  // 使用 tab 分隔列
                }
            }
            file << "\n";  // 换行分隔行

    // 关闭文件
    file.close();
}
void TorchMonitor::logCV_runtime(std::string& path){
    auto log_cv_tensor = torch_now_cvs;
    torch::Tensor cv_cpu_tensor = log_cv_tensor.to(torch::kCPU);
    torch::Tensor bias_cpu_tensor = this->torch_bias.to(torch::kCPU);
    torch::Tensor cpu_tensor = torch::cat({bias_cpu_tensor.squeeze(0).reshape({-1}),cv_cpu_tensor},0).reshape({-1});

    // 创建一个 TensorAccessor 对象来访问张量数据
    auto tensor_data = cpu_tensor.accessor<double, 1>();
    // 打开文件
    std::string log_name = path;

    std::ofstream file(log_name,std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " <<log_name<< std::endl;
        return;
    }
    // 写入张量数据
    for (int j = 0; j < cpu_tensor.size(0); ++j) {
                file << tensor_data[j];
                if (j < cpu_tensor.size(0) - 1) {
                    file << "\t\t";  // 使用 tab 分隔列
                }
            }
            file << "\n";  // 换行分隔行

    // 关闭文件
    file.close();
}



void TorchMonitor::box_to_tri(Box& box){
      //   if (box.triclinic == 0) {
      //   cpu_b_vector[0] = box.cpu_h[0];
      //   cpu_b_vector[1] = 0.0;
      //   cpu_b_vector[2] = 0.0;
      //   cpu_b_vector[3] = 0.0;
      //   cpu_b_vector[4] = box.cpu_h[1];
      //   cpu_b_vector[5] = 0.0;
      //   cpu_b_vector[6] = 0.0;
      //   cpu_b_vector[7] = 0.0;
      //   cpu_b_vector[8] = box.cpu_h[2];
      // } else {
        cpu_b_vector[0] = box.cpu_h[0];
        cpu_b_vector[1] = box.cpu_h[3];
        cpu_b_vector[2] = box.cpu_h[6];
        cpu_b_vector[3] = box.cpu_h[1];
        cpu_b_vector[4] = box.cpu_h[4];
        cpu_b_vector[5] = box.cpu_h[7];
        cpu_b_vector[6] = box.cpu_h[2];
        cpu_b_vector[7] = box.cpu_h[5];
        cpu_b_vector[8] = box.cpu_h[8];
      // }
}
// #endif

namespace {

void trim_monitor_config_value(std::string& value)
{
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    value.clear();
    return;
  }
  const auto last = value.find_last_not_of(" \t\r\n");
  value = value.substr(first, last - first + 1);
}

void unquote_monitor_config_value(std::string& value)
{
  if (value.size() >= 2 &&
      ((value.front() == '"' && value.back() == '"') ||
       (value.front() == '\'' && value.back() == '\''))) {
    value = value.substr(1, value.size() - 2);
  }
}

torch::Tensor cv_monitor_nl_to_side_array(torch::Tensor& torch_nl, int n_atoms)
{
  const int max_neighbors = static_cast<int>(torch_nl.numel()) / n_atoms;
  torch::Tensor starts = torch::arange(
    0,
    n_atoms,
    torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA));
  starts = starts.repeat({max_neighbors});
  const torch::Tensor mask = torch_nl >= 0;
  starts = starts.masked_select(mask);
  torch::Tensor ends = torch_nl.masked_select(mask).to(torch::kLong);
  return torch::stack({starts, ends}, 1);
}

static __global__ void find_neighbor_list_large_box_cv_monitor(
  const float neighbor_rc,
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
  int count_radial = 0;

  int cell_id;
  int cell_id_x;
  int cell_id_y;
  int cell_id_z;
  find_cell_id(
    box,
    x1,
    y1,
    z1,
    2.0f / neighbor_rc,
    nx,
    ny,
    nz,
    cell_id_x,
    cell_id_y,
    cell_id_z,
    cell_id);

  const int z_lim = box.pbc_z ? 2 : 0;
  const int y_lim = box.pbc_y ? 2 : 0;
  const int x_lim = box.pbc_x ? 2 : 0;

  for (int zz = -z_lim; zz <= z_lim; ++zz) {
    for (int yy = -y_lim; yy <= y_lim; ++yy) {
      for (int xx = -x_lim; xx <= x_lim; ++xx) {
        int neighbor_cell = cell_id + zz * nx * ny + yy * nx + xx;
        if (cell_id_x + xx < 0) {
          neighbor_cell += nx;
        }
        if (cell_id_x + xx >= nx) {
          neighbor_cell -= nx;
        }
        if (cell_id_y + yy < 0) {
          neighbor_cell += ny * nx;
        }
        if (cell_id_y + yy >= ny) {
          neighbor_cell -= ny * nx;
        }
        if (cell_id_z + zz < 0) {
          neighbor_cell += nz * ny * nx;
        }
        if (cell_id_z + zz >= nz) {
          neighbor_cell -= nz * ny * nx;
        }

        const int number_in_cell = cell_count[neighbor_cell];
        const int first_atom = cell_count_sum[neighbor_cell];
        for (int m = 0; m < number_in_cell; ++m) {
          const int n2 = cell_contents[first_atom + m];
          if (n1 == n2) {
            continue;
          }

          double dx = x[n2] - x1;
          double dy = y[n2] - y1;
          double dz = z[n2] - z1;
          apply_mic(box, dx, dy, dz);
          const float distance_squared =
            static_cast<float>(dx * dx + dy * dy + dz * dz);
          if (distance_squared >= neighbor_rc * neighbor_rc) {
            continue;
          }
          nl_radial[count_radial++ * n_atoms + n1] = n2;
        }
      }
    }
  }
  nn_radial[n1] = count_radial;
}

} // namespace

CVMonitorConfig CVMonitorConfig::from_file(const std::string& file_path)
{
  std::ifstream file(file_path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open Monitor config: " + file_path);
  }

  std::unordered_map<std::string, std::string> values;
  std::string line;
  while (std::getline(file, line)) {
    const size_t comment = line.find('#');
    if (comment != std::string::npos) {
      line = line.substr(0, comment);
    }
    const size_t colon = line.find(':');
    if (colon == std::string::npos) {
      continue;
    }
    std::string key = line.substr(0, colon);
    std::string value = line.substr(colon + 1);
    trim_monitor_config_value(key);
    trim_monitor_config_value(value);
    if (!key.empty()) {
      values[key] = value;
    }
  }

  CVMonitorConfig config;
  try {
    if (values.count("cv_log_interval")) {
      config.cv_log_interval = std::stoi(values.at("cv_log_interval"));
    }
    if (values.count("neighbor_rc")) {
      config.neighbor_rc = std::stod(values.at("neighbor_rc"));
    }
    if (values.count("max_neighbors")) {
      config.max_neighbors = std::stoi(values.at("max_neighbors"));
    }
    if (values.count("cv_log_path")) {
      config.cv_log_path = values.at("cv_log_path");
      unquote_monitor_config_value(config.cv_log_path);
    }
  } catch (const std::exception& error) {
    throw std::runtime_error(
      "Invalid Monitor config " + file_path + ": " + error.what());
  }

  if (config.cv_log_interval <= 0) {
    throw std::runtime_error("Monitor cv_log_interval must be positive.");
  }
  if (config.neighbor_rc <= 0.0) {
    throw std::runtime_error("Monitor neighbor_rc must be positive.");
  }
  if (config.max_neighbors <= 0) {
    throw std::runtime_error("Monitor max_neighbors must be positive.");
  }
  if (config.cv_log_path.empty()) {
    throw std::runtime_error("Monitor cv_log_path must not be empty.");
  }
  return config;
}

void CVMonitorConfig::print() const
{
  std::cout << "[Monitor] CV Log Interval: " << cv_log_interval << std::endl;
  std::cout << "[Monitor] CV Log Path: " << cv_log_path << std::endl;
  std::cout << "[Monitor] Neighbor RC: " << neighbor_rc << std::endl;
  std::cout << "[Monitor] Max Neighbors: " << max_neighbors << std::endl;
}

TorchCVMonitor::TorchCVMonitor(
  std::string model_path,
  std::string config_path,
  int n_atoms)
  : config_(CVMonitorConfig::from_file(config_path)), n_atoms_(n_atoms), cpu_box_(9)
{
  if (n_atoms_ <= 0) {
    throw std::runtime_error("Monitor requires at least one atom.");
  }
  try {
    torch::jit::setGraphExecutorOptimize(true);
    model_ = torch::jit::load(model_path, torch::kCUDA);
    model_.eval();
  } catch (const c10::Error& error) {
    std::cerr << "Error loading Monitor model " << model_path << ": " << error.what()
              << std::endl;
    throw;
  }

  cell_count_.resize(n_atoms_);
  cell_count_sum_.resize(n_atoms_);
  cell_contents_.resize(n_atoms_);
  nn_radial_.resize(n_atoms_);
  nl_radial_.resize(n_atoms_ * config_.max_neighbors);

  std::cout << "[Monitor] CVonly model loaded successfully from " << model_path << std::endl;
  config_.print();
  torch::cuda::synchronize();
}

std::unique_ptr<TorchCVMonitor> TorchCVMonitor::parse_monitor(
  const char** param,
  int num_param,
  int number_of_atoms)
{
  if (num_param != 3) {
    throw std::runtime_error(
      "Error parsing Monitor: expected \"Monitor model.pt config.yaml\".");
  }
  return std::make_unique<TorchCVMonitor>(param[1], param[2], number_of_atoms);
}

void TorchCVMonitor::compute_large_box(
  Box& box,
  const GPU_Vector<double>& position_per_atom)
{
  constexpr int block_size = 64;
  const int grid_size = (n_atoms_ - 1) / block_size + 1;
  const double cell_list_cutoff = 0.5 * config_.neighbor_rc;
  int number_of_bins[3];
  box.get_num_bins(cell_list_cutoff, number_of_bins);

  find_cell_list(
    cell_list_cutoff,
    number_of_bins,
    box,
    position_per_atom,
    cell_count_,
    cell_count_sum_,
    cell_contents_);

  find_neighbor_list_large_box_cv_monitor<<<grid_size, block_size>>>(
    config_.neighbor_rc,
    n_atoms_,
    number_of_bins[0],
    number_of_bins[1],
    number_of_bins[2],
    box,
    cell_count_.data(),
    cell_count_sum_.data(),
    cell_contents_.data(),
    position_per_atom.data(),
    position_per_atom.data() + n_atoms_,
    position_per_atom.data() + 2 * n_atoms_,
    nn_radial_.data(),
    nl_radial_.data());
  torch::cuda::synchronize();
}

void TorchCVMonitor::get_neighbor_list(
  Box& box,
  const GPU_Vector<double>& position_per_atom)
{
  nn_radial_.fill(-1);
  nl_radial_.fill(-1);
  compute_large_box(box, position_per_atom);
}

void TorchCVMonitor::box_to_tri(Box& box)
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

torch::Tensor TorchCVMonitor::from_cuda_memory(double* d_array, int size) const
{
  return torch::from_blob(
    d_array,
    {size},
    torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA));
}

torch::Tensor TorchCVMonitor::predict_cv(
  const torch::Dict<std::string, torch::Tensor>& inputs)
{
  auto result = model_.forward({inputs}).toGenericDict();
  torch::Dict<std::string, torch::Tensor> outputs;
  for (const auto& item : result) {
    outputs.insert(item.key().toStringRef(), item.value().toTensor());
  }
  if (!outputs.contains("cv_now")) {
    throw std::runtime_error(
      "Monitor model output must contain cv_now (pygas CVOnlyProcedure contract).");
  }
  torch::Tensor cv = outputs.at("cv_now").reshape({-1}).to(torch::kFloat64);
  if (cv.numel() == 0) {
    throw std::runtime_error("Monitor model cv_now output must not be empty.");
  }
  if (!cv.is_cuda()) {
    cv = cv.to(torch::kCUDA);
  }
  return cv;
}

void TorchCVMonitor::log_cv() const
{
  torch::Tensor cv_cpu = torch_now_cvs_.to(torch::kCPU).contiguous().reshape({-1});
  const auto values = cv_cpu.accessor<double, 1>();
  std::ofstream file(config_.cv_log_path, std::ios::app);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open Monitor CV log file: " + config_.cv_log_path);
  }
  for (int index = 0; index < cv_cpu.size(0); ++index) {
    if (index != 0) {
      file << "\t\t";
    }
    file << values[index];
  }
  file << '\n';
}

void TorchCVMonitor::process(Box& box, const GPU_Vector<double>& positions, int step)
{
  if (step < 0) {
    throw std::runtime_error("Monitor step must not be negative.");
  }
  if (step % config_.cv_log_interval != 0) {
    return;
  }
  if (positions.size() != 3 * n_atoms_) {
    throw std::runtime_error("Monitor atom count is inconsistent with model.xyz.");
  }

  box_to_tri(box);
  get_neighbor_list(box, positions);

  torch::Tensor positions_tensor =
    from_cuda_memory(const_cast<double*>(positions.data()), positions.size())
      .detach()
      .clone()
      .reshape({3, -1})
      .transpose(0, 1);
  torch::Tensor cell_tensor = torch::from_blob(
    cpu_box_.data(), {9}, torch::TensorOptions().dtype(torch::kFloat64))
                                .to(torch::kCUDA)
                                .reshape({3, 3});
  torch_nl_ = torch::from_blob(
    nl_radial_.data(),
    {n_atoms_ * config_.max_neighbors},
    torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA));

  torch::Dict<std::string, torch::Tensor> inputs;
  inputs.insert("positions", positions_tensor);
  inputs.insert("cell", cell_tensor);
  inputs.insert("side_array", cv_monitor_nl_to_side_array(torch_nl_, n_atoms_));
  torch_now_cvs_ = predict_cv(inputs);
  torch::cuda::synchronize();
  log_cv();
}
