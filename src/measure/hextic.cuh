/*
    Copyright 2017 Zheyong Fan, Ville Vierimaa, Mikko Ervasti, and Ari Harju
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once
#include "utilities/gpu_vector.cuh"
#include <vector>

class Box;
class Hextic
{
public:
  bool compute_ = false;
  void parse(const char** param, const int num_param);
  void preprocess(const int num_atoms);
  void process(
    const int step,
    Box& box,
    GPU_Vector<int>& type,
    GPU_Vector<double>& position_per_atom,
    std::vector<double>& cpu_position_per_atom);
  void postprocess();

private:
  int sample_interval_ = 1;
  double cutoff_;
  double correlation_length_;
  int mesh_=1000;
  GPU_Vector<double> radius_;
  int num_atoms_;
  GPU_Vector<int> NN;
  GPU_Vector<int> NL;
  GPU_Vector<int> cell_count;
  GPU_Vector<int> cell_count_sum;
  GPU_Vector<int> cell_contents;
  GPU_Vector<int> order_count;
  GPU_Vector<double> orien_order;
  GPU_Vector<double> sum_cos, sum_sin, phi;
  std::vector<double> cpu_cos, cpu_sin, cpu_phi;
  std::vector<double> cpu_orien_order;
  std::vector<int> cpu_order_count;
  void initialize_parameters(const int num_atoms);
  void allocate_memory();
  void output_phi(std::vector<double>& cpu_position_per_atom);
  void output_order_correlation();
};
