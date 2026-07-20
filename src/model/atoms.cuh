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

extern cublasHandle_t cublashandle;

void print_arr(double* a, size_t size,const char* name="");

void print_arr(int* a, size_t size,const char* name="");

void print_gpu(GPU_Vector<int>& a, const char* name="");

void print_gpu(GPU_Vector<double>& a, const char* name="");

void print_gpu(double* a, int size, const char* name="");

class BaseAtoms;
class Atoms;
class VCWrapper;
class RotationFreeVCWrapper;


class BaseAtoms
{
protected:
  int natoms;
  GPU_Vector<double> positions; // size: (natoms, 3)
  GPU_Vector<double> potential_per_atom; // size: (natoms)
  GPU_Vector<double> forces; // size: (natoms, 3)

public:
  bool is_cell_filter = false;
  Force* p_force;
  virtual void compute() = 0;

  virtual GPU_Vector<double>& get_positions() { return positions;}
  
  virtual GPU_Vector<double>& get_potential_per_atom(){
    return potential_per_atom;
  }

  virtual double get_energy() = 0;
  
  virtual GPU_Vector<double>& get_forces() {
    return forces;
  }

  virtual bool has_cell_degrees_of_freedom() const { return false; }

  virtual int get_real_atom_count_per_block() const { return natoms; }

  virtual int get_atoms_per_block() const { return natoms; }

  virtual bool update_minimizer_force_max(double force_max) { return false; }

  // virtual void set_box(GPU_Vector<double> h0);

  virtual void set_calc(Force& force){
    printf("set calc\n");
    p_force = &force;
  };
};

class Atoms: public BaseAtoms
{
friend class VCWrapper;
friend class RotationFreeVCWrapper;

public:
  bool is_cell_filter = false;
  Box box;
  // std::vector<int> cpu_type;
  std::vector<std::string> cpu_atom_symbol; // symbol strings
  std::vector<Group> group;
  // std::vector<double> cpu_positions;
  GPU_Vector<int> type; // size: (natoms), type(int) of each atom
  // GPU_Vector<double> masses;
  // double *h; // 18 elements, first 9 are cell, last 9 are the inverse of cell.
  GPU_Vector<double> virials; // size: (natoms, 9)

  Atoms();

  Atoms(const Atoms* p_atoms0, double* new_position);

  Atoms(const Atoms& atoms0);

  Atoms(Atoms&&) = default;

  Atoms& operator=(Atoms&&) = default;
  
  Atoms(
    Force& force0,
    Box& box0,
    GPU_Vector<double>& positions0,
    GPU_Vector<int>& type0,
    std::vector<Group>& group0,
    GPU_Vector<double>& potential_per_atom0,
    GPU_Vector<double>& forces0,
    GPU_Vector<double>& virials0);

  Atoms(
    Force& force0,
    Box& box0,
    GPU_Vector<double>& positions0,
    std::vector<std::string> cpu_atom_symbol0,
    GPU_Vector<int>& type0,
    std::vector<Group>& group0,
    GPU_Vector<double>& potential_per_atom0,
    GPU_Vector<double>& forces0,
    GPU_Vector<double>& virials0);

  // Atoms(Atom& atom, std::vector<Group>& group0);

  Atoms(const char* filename);

  Atoms(std::ifstream& input, bool& success, bool print_flag = true);

  ~Atoms();

  void initialize(Atom& atom);

  // int number_of_type(std::string& symbol);

  void compute();

  double get_energy();

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
protected:
  VCWrapper() = default;

  double* virial = nullptr; // size: 9, managed memory

private:
  void build_VCWrapper(std::vector<double> p, double* h_ref0, double cell_factor0=-1.0);

public:
  bool is_cell_filter = true;
  double cell_factor = 1.0;
  std::vector<double> pressure = std::vector<double>(9,0.0);
  double* h_ref = nullptr; // size: 18, managed memory. first 9 are reference cell, last 9 are the inverse.
  std::unique_ptr<Atoms> p_atoms;
  double* deform = nullptr; // size: 18, managed memory. first 9 are deform, last 9 are the inverse.
  GPU_Vector<double> d_h; // size: 18, device memory


  VCWrapper(Atoms& atoms, std::vector<double> p, double* h_ref0, double cell_factor0=-1.0);
  VCWrapper(Atoms& atoms, std::vector<double> p, double cell_factor0=-1.0);

  VCWrapper(const char* filename, std::vector<double> p, double* h_ref0, double cell_factor0=-1.0);
  VCWrapper(std::ifstream& input, bool& success, std::vector<double> p, double* h_ref0, double cell_factor0=-1.0);
  VCWrapper(std::ifstream& input, bool& success, std::vector<double> p, double cell_factor0=-1.0);

  VCWrapper(const VCWrapper& atoms0, double* new_position);
  VCWrapper(Atoms* p_atoms0, double* new_position);

  ~VCWrapper();

  void initialize(int natoms0);

  void set_calc(Force& force);

  void compute();

  double get_energy();

  GPU_Vector<double>& get_potential_per_atom();

  // from positions and box of atoms to build vcwrapper positions
  virtual GPU_Vector<double>& build_positions();

  // use updated vcwrapper positions to reset atoms positions and box
  void set_positions();

  Atoms* get_p_atoms() {return p_atoms.get();}

  // void set_box(Box& box0);

  void compute_deform();

  bool has_cell_degrees_of_freedom() const override { return true; }

  int get_real_atom_count_per_block() const override { return natoms - 3; }

  int get_atoms_per_block() const override { return natoms; }

};

class RotationFreeVCWrapper: public VCWrapper
{
// the cell part coordinates are defined as 1/2*D D^T, the according forces are D^(-T) V D^(-1)
public:
  bool is_cell_filter = true;
  RotationFreeVCWrapper(Atoms& atoms, std::vector<double> p, double* h_ref0, double cell_factor0=-1.0);
  RotationFreeVCWrapper(Atoms& atoms, std::vector<double> p, double cell_factor0=-1.0);

  RotationFreeVCWrapper(const char* filename, std::vector<double> p, double* h_ref0, double cell_factor0=-1.0);
  RotationFreeVCWrapper(std::ifstream& input, bool& success, std::vector<double> p, double* h_ref0, double cell_factor0=-1.0);
  RotationFreeVCWrapper(std::ifstream& input, bool& success, std::vector<double> p, double cell_factor0=-1.0);

  RotationFreeVCWrapper(const RotationFreeVCWrapper& atoms0, double* new_position);
  RotationFreeVCWrapper(Atoms* p_atoms0, double* new_position);

  void compute();

  GPU_Vector<double>& build_positions() override;

  void set_positions();
};

void save_one_frame(
  FILE* fid_,
  const Box& box,
  double energy,
  double enthalpy,
  const std::vector<std::string>& cpu_atom_symbol,
  GPU_Vector<double>& position_per_atom,
  std::vector<double>& cpu_position_per_atom);

void save_one_frame(
  FILE* fid_,
  const Box& box,
  double energy,
  double enthalpy,
  const std::vector<std::string>& cpu_atom_symbol,
  GPU_Vector<double>& position_per_atom);


void save_xyz_virials(
  const Box& box,
  const std::vector<std::string>& cpu_atom_symbol,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& virial_per_atom);
