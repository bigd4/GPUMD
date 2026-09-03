#include "main_sampling/gas-model-runner.cuh"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv)
{
  if (argc < 7 || argc > 8) {
    std::cerr << "Usage: gas-aoti-smoke MODEL.pt2 ROUTE N_ATOMS CV_SIZE MAX_EDGES "
                 "MAX_HISTORY [OPES_CAPACITY]\n";
    return 2;
  }
  try {
    const std::string path = argv[1];
    const std::string route = argv[2];
    const int n_atoms = std::stoi(argv[3]);
    const int cv_size = std::stoi(argv[4]);
    const int max_edges = std::stoi(argv[5]);
    const int max_history = std::stoi(argv[6]);
    const int opes_capacity = argc == 8 ? std::stoi(argv[7]) : 0;
    GASModelRunner runner(
      path, route, n_atoms, cv_size, max_edges, max_history, opes_capacity);
    const auto doubles =
      torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA);
    torch::Dict<std::string, torch::Tensor> inputs;
    inputs.insert("positions", torch::zeros({n_atoms, 3}, doubles));
    inputs.insert("cell", torch::eye(3, doubles));
    inputs.insert(
      "side_array",
      torch::empty(
        {0, 2}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA)));
    if (route == "bias" || route == "cvonly" || route == "bias_or_cvonly") {
      inputs.insert("cv_traj", torch::empty({0, cv_size}, doubles));
      inputs.insert("bias_traj", torch::empty({0}, doubles));
      inputs.insert(
        "saved_cv_nums",
        torch::zeros(
          {}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA)));
    } else if (route == "opes") {
      inputs.insert(
        "opes_step",
        torch::zeros(
          {}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA)));
      const auto state = runner.initialize_opes_state(opes_capacity);
      for (const auto& item : state) {
        inputs.insert(item.key(), item.value());
      }
    }
    const auto outputs = runner.run(inputs);
    std::cout << "AOTI_SMOKE_OK backend=" << runner.backend()
              << " route=" << runner.route() << " outputs=" << outputs.size();
    if (outputs.contains("bias")) {
      std::cout << " bias=" << outputs.at("bias").item<double>();
    }
    if (outputs.contains("status")) {
      std::cout << " status=" << outputs.at("status").item<int64_t>();
    }
    std::cout << std::endl;
  } catch (const std::exception& error) {
    std::cerr << "AOTI_SMOKE_FAILED: " << error.what() << std::endl;
    return 1;
  }
  return 0;
}
