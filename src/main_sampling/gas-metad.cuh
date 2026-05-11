// #ifdef USE_GAS
#pragma once
#include "force/potential.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_vector.cuh"
#include "force/neighbor.cuh"
#include <torch/torch.h>
#include <torch/script.h>
#include <unordered_map>
#include <string>
#include <stdexcept>
#include <unordered_map>
#include <iostream>
#include <fstream>
#include <sstream>

// 定义一个结构体用于存储配置数据
struct Config {
    int max_cv_nums = 10000;
    int cv_size = 1;
    int cv_storage_interval = 100;
    int cv_change_interval = 100;
    int cv_log_interval = 100;
    double neighbor_rc = 5.0;
    int max_neighbors = 256;
    int n_atoms = 0;
    bool is_opt_=0;
    bool is_obs_=0;
    int debug_interval=0;
    

    // 打印结构体内容
    void print() const {
        std::cout << "[GAScfg]Max CV Nums: " << max_cv_nums << std::endl;
        std::cout << "[GAScfg]CV Size: " << cv_size << std::endl;
        std::cout << "[GAScfg]CV Storage Interval: " << cv_storage_interval << std::endl;
        std::cout << "[GAScfg]CV Change Interval: " << cv_change_interval << std::endl;
        std::cout << "[GAScfg]CV Log Interval: " << cv_log_interval << std::endl;
        std::cout << "[GAScfg]Neighbor RC: " << neighbor_rc << std::endl;
        std::cout << "[GAScfg]Max Neighbors: " << max_neighbors << std::endl;
        std::cout << "[GAScfg]Number of Atoms: " << n_atoms << std::endl;
        std::cout << "[GAScfg]Debug Interval: " << debug_interval << std::endl;
        std::cout << "[GAScfg]MetaCell Enabled: " << is_opt_ << std::endl;
        std::cout << "[GAScfg]Observe Enabled: " << is_obs_ << std::endl;
    }

    // 从文件初始化结构体的静态方法
    static Config fromFile(const std::string& filePath) {
        Config base_config;
        return fromFile(filePath, base_config);
    }

    static Config fromFile(const std::string& filePath, const Config& base_config) {
        std::unordered_map<std::string, std::string> configMap;
        std::ifstream file(filePath);

        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + filePath);
        }

        std::string line;
        while (std::getline(file, line)) {
            // 跳过空行和注释行
            if (line.empty() || line[0] == '#') {
                continue;
            }

            // 查找冒号分隔符
            size_t colon_pos = line.find(':');
            if (colon_pos == std::string::npos) {
                continue; // 跳过不符合格式的行
            }

            // 提取键和值，并去掉多余的空格
            std::string key = line.substr(0, colon_pos);
            std::string value = line.substr(colon_pos + 1);

            // 去除行内注释
            size_t comment_pos = value.find('#');
            if (comment_pos != std::string::npos) {
                value = value.substr(0, comment_pos);
            }

            // 修剪键和值的空白字符
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);

            // 存储到哈希表
            configMap[key] = value;
        }

        file.close();

        // 先使用默认/基础配置，再按 YAML 覆盖
        Config config = base_config;
        try {
            if (configMap.find("max_cv_nums") != configMap.end()) {
                config.max_cv_nums = std::stoi(configMap.at("max_cv_nums"));
            }
            if (configMap.find("cv_size") != configMap.end()) {
                config.cv_size = std::stoi(configMap.at("cv_size"));
            }
            if (configMap.find("cv_storage_interval") != configMap.end()) {
                config.cv_storage_interval = std::stoi(configMap.at("cv_storage_interval"));
            }
            if (configMap.find("cv_change_interval") != configMap.end()) {
                config.cv_change_interval = std::stoi(configMap.at("cv_change_interval"));
            }
            if (configMap.find("cv_log_interval") != configMap.end()) {
                config.cv_log_interval = std::stoi(configMap.at("cv_log_interval"));
            }
            if (configMap.find("neighbor_rc") != configMap.end()) {
                config.neighbor_rc = std::stod(configMap.at("neighbor_rc"));
            }
            if (configMap.find("max_neighbors") != configMap.end()) {
                config.max_neighbors = std::stoi(configMap.at("max_neighbors"));
            }
            if (configMap.find("n_atoms") != configMap.end()) {
                config.n_atoms = std::stoi(configMap.at("n_atoms"));
            }
            if (configMap.find("debug_interval") != configMap.end()) {
                config.debug_interval = std::stoi(configMap.at("debug_interval"));
            }

            // flag
            config.is_opt_ = (configMap.find("MetaCell")!=configMap.end());
            config.is_obs_ = (configMap.find("Observe")!=configMap.end());


        } catch (const std::exception& ex) {
            throw std::runtime_error("Error parsing configuration: " + std::string(ex.what()));
        }



        return config;
    }
};

struct TorchMetad : public Potential
{
public:
    TorchMetad(std::string model_path,std::string cfg_path,int n_atoms);
    TorchMetad(std::string model_path,std::string cfg_path,std::string gaussian_path,int n_atoms);
    TorchMetad(int n_atoms);
    void compute_large_box(Box& box,const GPU_Vector<double>& position_per_atom);
    void get_neighbor_list(Box& box,const GPU_Vector<double>& position_per_atom);

    void compute(
        Box& box,
        const GPU_Vector<int>& type,
        const GPU_Vector<double>& positions,
        GPU_Vector<double>& potentials,
        GPU_Vector<double>& forces,
        GPU_Vector<double>& virial);

    void compute(
    const float temperature,
    Box& box,
    const GPU_Vector<int>& type,
    const GPU_Vector<double>& positions,
    GPU_Vector<double>& potentials,
    GPU_Vector<double>& forces,
    GPU_Vector<double>& virial){
       compute(box, type, positions, potentials, forces, virial);
       // enable this if opes needed.
       };

    torch::Dict<std::string, torch::Tensor> predict(const torch::Dict<std::string, torch::Tensor>& inputs);
    torch::Tensor _FromCudaMemory(double* d_array, int size);
    void box_to_tri(Box& box);
    void logCV_runtime(void);
    void logCV_runtime(std::string& path);
    void appendCVtoTraj(bool is_opt);

    static std::unique_ptr<TorchMetad> parse_GASMD(const char** param, int num_param, const int number_of_atoms) {
       if(num_param==1)
       {
        return std::make_unique<TorchMetad>(number_of_atoms);
       }
       else if (num_param==2)
       {
        throw std::runtime_error("Error parsing GASMD: params shapes like \"GASMD model.pt cfg.yaml gaussian.txt\", but found "+ std::string(param[0])+std::string(param[1]));
       }
       else if (num_param==3)
       {
        std::string model_path = param[1];
        std::string cfg_path = param[2];
        return std::make_unique<TorchMetad>(model_path,cfg_path,number_of_atoms);
       }
       else if (num_param==4)
       {
        std::string model_path = param[1];
        std::string cfg_path = param[2];
        std::string gaussian_path =param[3];
        return std::make_unique<TorchMetad>(model_path,cfg_path,gaussian_path,number_of_atoms);
       }
       else{
        throw std::runtime_error("Error parsing GASMD: params shapes like \"GASMD model.pt cfg.yaml\"");
       }
       
    }

    Config config;

private:

    bool is_opt=0;
    bool is_obs=0;

    int saved_cv_nums=0;
    int now_step = 0;
    int n_atoms_=1;
    int debug_interval=0;
    torch::Tensor torch_now_cvs;
    torch::Tensor torch_delta_cv_save;
    torch::Tensor torch_cv_traj;
    torch::Tensor torch_bias_traj;
    torch::Tensor torch_bias;
    torch::Tensor torch_NN;
    torch::Tensor torch_NL;
    // 存 cell 参数
    // GPU_Vector<double> gpu_v_vector;  // Total Virial (GPU)
    // GPU_Vector<double> gpu_v_factor;  // Scaling factor of the virial (GPU)
    std::vector<double> cpu_b_vector; // Box
    // GPU_Vector<double> gpu_b_vector;  // Box (GPU)
    // 找近邻
    GPU_Vector<int> cell_count;
    GPU_Vector<int> cell_count_sum;
    GPU_Vector<int> cell_contents;
    GPU_Vector<int> NN_radial;    // radial neighbor list
    GPU_Vector<int> NL_radial;    // radial neighbor list
    // torch-model
    torch::jit::script::Module model; // TorchScript 模型
    torch::Tensor mean_bias_force;

    // name
    std::string gaussian_name = "GASGaussian.txt";
};

// #endif
