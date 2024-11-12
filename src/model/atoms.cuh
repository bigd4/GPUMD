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
#include "utilities/common.cuh"
#include "force/force.cuh"
#include "read_xyz.cuh"
#include "atom.cuh"
// #include "group.cuh"
#include <vector>
// #include <cstring>
#include <cublas_v2.h>
#include <cmath>
// #include <cuda_runtime.h>
using namespace std;

class BaseAtoms;
class Atoms;
class VCWrapper;


class BaseAtoms
{
protected:
  int natoms;
  GPU_Vector<double> positions;
  GPU_Vector<double> potential_per_atom;
  GPU_Vector<double> forces;

public:
  Force* p_force;
  virtual void compute() = 0;

  virtual GPU_Vector<double>& get_positions() { return positions;}
  
  virtual GPU_Vector<double>& get_potential_per_atom(){
    return potential_per_atom;
  }
  
  virtual GPU_Vector<double>& get_forces() {
    return forces;
  }

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
  // cublasHandle_t handle;

public:
  Box box;
  // vector<int> cpu_type;
  vector<string> cpu_atom_symbol;
  vector<Group> group;
  // vector<double> cpu_positions;
  GPU_Vector<int> type;
  // GPU_Vector<double> masses;
  // vector<GPU_Vector<double>> velocities;
  // std::vector<GPU_Vector<double>> forces;
  // std::vector<GPU_Vector<double>> virials;
  // double *h; // 18 elements, first 9 are cell, last 9 are the inverse of cell.
  GPU_Vector<double> virials;
  // double *test;

  Atoms();

  Atoms(const Atoms& atoms0, double* new_position);

  Atoms(const Atoms& atoms0);

  Atoms(Atoms&&) = default;

  Atoms& operator=(Atoms&&) = default;
  
  Atoms(
    Force& force0,
    Box& box0,
    GPU_Vector<double>& positions0,
    GPU_Vector<int>& type0,
    vector<Group>& group0,
    GPU_Vector<double>& potential_per_atom0,
    GPU_Vector<double>& forces0,
    GPU_Vector<double>& virials0);

  Atoms(
    Force& force0,
    Box& box0,
    GPU_Vector<double>& positions0,
    vector<string> cpu_atom_symbol0,
    GPU_Vector<int>& type0,
    vector<Group>& group0,
    GPU_Vector<double>& potential_per_atom0,
    GPU_Vector<double>& forces0,
    GPU_Vector<double>& virials0);

  Atoms(Atom& atom, vector<Group>& group0);

  Atoms(const char* filename);

  ~Atoms();

  void initialize(Atom& atom);

  // int number_of_type(string& symbol);

  void compute();

  virtual double get_energy();

  // GPU_Vector<double>& get_positions();
  
  virtual void set_positions(){};
  
  // GPU_Vector<double>& get_potential_per_atom();

  virtual int get_natoms() {return natoms;}

  virtual Atoms* get_p_atoms() {return this;}

  virtual void set_box(Box& box0);

  virtual void set_box(double* cpu_h, int len);

  virtual void set_box(GPU_Vector<double>& h0, int size);

};

class VCWrapper: public Atoms
{
private:
  double* virial;

public:
  double cell_factor = 1.0;
  vector<double> pressure = vector<double>(9,0.0);
  double* ref_h; // 18 elements, first 9 are reference cell, last 9 are the inverse.
  Atoms* p_atoms;
  double* deform; // 18 elements, first 9 are deform, last 9 are the inverse.
  GPU_Vector<double> d_h;

  // VCWrapper(Atoms& atoms, double *p, int l_p, double* h0);
  // VCWrapper(Atoms& atoms, double* p, int l_p);

  VCWrapper(Atoms& atoms, vector<double> p, double* h0);
  VCWrapper(Atoms& atoms, vector<double> p);

  VCWrapper(const VCWrapper& atoms0, double* new_position);

  VCWrapper(Atoms* p_atoms0, double* new_position);

  ~VCWrapper();

  void initialize(int natoms0);

  void set_calc(Force& force);

  void compute();

  double get_energy();

  GPU_Vector<double>& get_potential_per_atom();

  // from positions and box of atoms to build vcwrapper positions
  GPU_Vector<double>& build_positions();

  // use updated vcwrapper positions to reset atoms positions and box
  void set_positions();

  Atoms* get_p_atoms() {return p_atoms;}

  // void set_box(Box& box0);

  void compute_deform();



};

// Atoms& xyz2atoms();

// void xyz2atoms(Atoms& atoms);

// Atoms xyz2atoms();

void print_arr(double* a, size_t size,const char* name="");

void print_arr(int* a, size_t size,const char* name="");

void print_gpu(GPU_Vector<int>& a, const char* name="");

void print_gpu(GPU_Vector<double>& a, const char* name="");

void print_gpu(double* a, int size, const char* name="");

void save_one_frame(
  FILE* fid_,
  const Box& box,
  double enthalpy,
  const std::vector<std::string>& cpu_atom_symbol,
  GPU_Vector<double>& position_per_atom,
  std::vector<double>& cpu_position_per_atom);

void save_one_frame(
  FILE* fid_,
  const Box& box,
  double enthalpy,
  const std::vector<std::string>& cpu_atom_symbol,
  GPU_Vector<double>& position_per_atom);