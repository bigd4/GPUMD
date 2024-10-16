/*
    Copyright 2017 Zheyong Fan and GPUMD development team
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
#include "utilities/read_file.cuh"
#include "model/read_xyz.cuh"
#include "force/force.cuh"
#include "atom.cuh"
// #include "group.cuh"
#include <vector>
// #include <cstring>
using namespace std;

class Atoms
{
public:
  int natoms = 0;
  double energy = 0.0;
  Force* p_force;
  Box box;
  vector<string> cpu_atom_symbol;
  vector<Group> group;
  GPU_Vector<int> type;
  GPU_Vector<double> positions;
  // GPU_Vector<double> masses;
  GPU_Vector<double> potential_per_atom;
  GPU_Vector<double> forces;
  GPU_Vector<double> virials;
  // vector<GPU_Vector<double>> velocities;
  // std::vector<GPU_Vector<double>> forces;
  // std::vector<GPU_Vector<double>> virials;

  Atoms(){};

  Atoms(const Atoms& atoms0);

  Atoms(
  Box& _box,
  GPU_Vector<double>& _positions,
  GPU_Vector<int>& _type,
  vector<Group>& _group,
  GPU_Vector<double>& _potentials,
  GPU_Vector<double>& _forces,
  GPU_Vector<double>& _virials);

  Atoms(Atom& atom);


  int number_of_type(string& symbol);
  void compute();

  GPU_Vector<double>& get_forces();


  GPU_Vector<double>& get_virial();

  void set_positions();

  void set_box();

};

class VCWrapper
{
  VCWrapper(){};
};

Atoms xyz2atoms();