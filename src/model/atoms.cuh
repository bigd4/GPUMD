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
#include <cublas_v2.h>
// #include <cuda_runtime.h>
using namespace std;

class BaseAtoms;
class Atoms;
class VCWrapper;


class BaseAtoms
{
protected:
  // bool changed_after_last_compute = true;
  GPU_Vector<double> positions;
  GPU_Vector<double> potential_per_atom;
  GPU_Vector<double> forces;
  // GPU_Vector<double> virials;

public:
  int natoms = 0;
  Force* p_force;
  virtual void compute() = 0;

  virtual GPU_Vector<double>& get_positions() { return positions;}

  virtual void set_positions(GPU_Vector<double>& positions0) { positions = move(positions0);}
  
  virtual GPU_Vector<double>& get_potential_per_atom(){
    // if (changed_after_last_compute) compute();
    return potential_per_atom;
  }

  virtual GPU_Vector<double>& get_forces() {
    // if (changed_after_last_compute) compute();
    return forces;
  }

  // virtual GPU_Vector<double>& get_virials() {
  //   // if (changed_after_last_compute) compute();
  //   return virials;
  // }

  // virtual GPU_Vector<double>& get_virial();
  // virtual double*& get_h() { return h; }


  // virtual void set_box(GPU_Vector<double> h0);

  virtual void set_calc(Force& force){
    printf("set calc\n");
    p_force = &force;
  };
};

class Atoms: public BaseAtoms
{
friend class VCWrapper;

protected:
  cublasHandle_t handle;

public:
  double energy = 0.0;
  Box box;
  vector<int> cpu_type;
  vector<string> cpu_atom_symbol;
  vector<Group> group;
  vector<double> cpu_positions;
  GPU_Vector<int> type;
  // GPU_Vector<double> masses;
  // vector<GPU_Vector<double>> velocities;
  // std::vector<GPU_Vector<double>> forces;
  // std::vector<GPU_Vector<double>> virials;
  // double *h; // 18 elements, first 9 are cell, last 9 are the inverse of cell.
  GPU_Vector<double> virials;
  double *test;

  Atoms();

  Atoms(const Atoms& atoms0) = default;

  Atoms(Atoms&&) = default;

  Atoms& operator=(Atoms&&) = default;

  ~Atoms();

  // Atoms(
  // Box& _box,
  // GPU_Vector<double>& _positions,
  // GPU_Vector<int>& _type,
  // vector<Group>& _group,
  // GPU_Vector<double>& _potentials,
  // GPU_Vector<double>& _forces,
  // GPU_Vector<double>& _virials);

  Atoms(Atom& atom, vector<Group>& group0);

  Atoms(const char* filename);

  void initialize(Atom& atom);

  // int number_of_type(string& symbol);

  void compute();

  // GPU_Vector<double>& get_positions();
  // void set_positions();
  
  // GPU_Vector<double>& get_potential_per_atom();

  // GPU_Vector<double>& get_forces();

  // GPU_Vector<double>& get_virials();

  // GPU_Vector<double>& get_virial();

  virtual void set_box(Box& box0);

  virtual void set_box(double* cpu_h, int len);

  virtual void set_box(GPU_Vector<double>& h0, int size);

};

class VCWrapper: public Atoms
{
private:
  double* virial;

public:
  int natoms = 0; // real_natoms + 3
  double cell_factor = 1.0;
  double pressure[9] = {0.0};
  double* ref_h; // 18 elements, first 9 are reference cell, last 9 are the inverse.
  Atoms* p_atoms;
  double* deform; // 18 elements, first 9 are deform, last 9 are the inverse.
  GPU_Vector<double> d_h;

  VCWrapper(Atoms& atoms, double *p, int l_p, double* h0);

  VCWrapper(Atoms& atoms, double *p, int l_p);

  ~VCWrapper();

  void initialize(int natoms0);

  void set_calc(Force& force);

  void compute();

  GPU_Vector<double>& build_positions();

  void set_positions();

  GPU_Vector<double>& get_forces();

  // void set_box(Box& box0);

  void compute_deform();

  // GPU_Vector<double>& get_stress();


};

// Atoms& xyz2atoms();

// void xyz2atoms(Atoms& atoms);

// Atoms xyz2atoms();