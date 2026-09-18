#pragma once
#include "io/output_format.h"
#include "model/box.cuh"
#include "model/group.cuh"
#include "utilities/gpu_vector.cuh"
#include <numeric>
#include <sstream>
#include <iomanip>
#include <locale>
namespace trajectory {
inline std::vector<int> selection(int n, const std::vector<Group>& groups, int method=-1, int id=-1) {
  if(method<0) {std::vector<int> indices(n);std::iota(indices.begin(),indices.end(),0);return indices;}
  const auto& g=groups.at(method);
  auto start=g.cpu_size_sum.at(id), size=g.cpu_size.at(id);
  return std::vector<int>(g.cpu_contents.begin()+start,g.cpu_contents.begin()+start+size);
}
inline std::string tensor(const std::string& name, const double* values, int n) {
  std::ostringstream s;s.imbue(std::locale::classic());s<<std::setprecision(17)<<' '<<name<<"=\"";
  for(int i=0;i<n;++i) {if(i)s<<' ';s<<values[i];}s<<'"';return s.str();
}
inline std::string geometry(const Box& box) {
  double cell[9];const int order[9]={0,3,6,1,4,7,2,5,8};
  for(int i=0;i<9;++i) cell[i]=box.cpu_h[order[i]];
  return tensor("Lattice",cell,9)+" pbc=\""+(box.pbc_x?"T":"F")+" "+(box.pbc_y?"T":"F")+" "+(box.pbc_z?"T":"F")+"\"";
}
inline std::string stamp(double time, int step, int method=-1, int id=-1) {
  std::ostringstream s;s.imbue(std::locale::classic());s<<std::setprecision(17)<<"Time="<<time<<" Step="<<step;
  if(method>=0) s<<" grouping_method="<<method<<" group_id="<<id;
  return s.str();
}
inline void add_groups(Frame& frame, const std::vector<Group>& groups, const std::vector<int>& indices) {
  if(groups.empty()) return;
  Array a{"group",groups.size(),'I',{},{},{}};
  for(int i:indices) for(const auto& g:groups) a.integer.push_back(g.cpu_label.at(i));
  frame.arrays.push_back(std::move(a));
}
std::string thermodynamics(GPU_Vector<double>& virial, GPU_Vector<double>& thermo);
} // namespace trajectory
