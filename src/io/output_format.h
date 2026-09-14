#pragma once
// C++14 boundary: no C++17 codec types leak into CUDA translation units.
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trajectory {
struct Config {
  std::string format = "xyz";
  bool lossless = false;
  int compression_level = 3;
  void parse(const std::vector<std::string>& tokens);
};
struct Array {
  std::string name;
  std::size_t components;
  char type;
  std::vector<double> real;
  std::vector<std::int64_t> integer;
  std::vector<std::string> strings;
};
struct Frame {
  std::size_t count = 0;
  std::string info;
  double cell[9] = {}; // used by read_model only
  bool pbc[3] = {};
  std::vector<Array> arrays;
  template<class T>
  void add_real(const std::string& name, int components, const std::vector<T>& values,
                std::size_t stride, const std::vector<int>& indices, double scale = 1.0,
                const int* order = nullptr) {
    Array a{name, static_cast<std::size_t>(components), 'R', {}, {}, {}};
    a.real.reserve(indices.size() * components);
    for(int i : indices) for(int d=0; d<components; ++d)
      a.real.push_back(values.at(i + stride * (order ? order[d] : d)) * scale);
    arrays.push_back(std::move(a));
  }
  void add_strings(const std::string& name, const std::vector<std::string>& values,
                   const std::vector<int>& indices);
};
std::string filename(const std::string& original, const Config& config);
bool is_binary(const std::string& path);
Frame read_model(const std::string& path);
class Output {
public:
  Output();
  ~Output();
  Output(const Output&) = delete;
  Output& operator=(const Output&) = delete;
  void configure(const Config& config);
  bool enabled() const { return config_.format != "xyz"; }
  void write(const std::string& path, const Frame& frame, bool append = true);
private:
  Config config_;
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace trajectory
