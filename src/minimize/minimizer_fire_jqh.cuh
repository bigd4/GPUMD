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
#include "force/force.cuh"
#include "minimizer.cuh"
#include "utilities/common.cuh"
#include "model/atoms.cuh"

class Minimizer_FIRE_JQH : public Minimizer
{
private:
  double f_inc = 1.1;
  double f_dec = 0.5;
  double alpha_start = 0.25;
  double f_alpha = 0.99;
  double dt_0 = 1 / TIME_UNIT_CONVERSION; // Time step of 1 fs.
  double dt_max = 1 * dt_0;
  double dt_min = 0.02 * dt_0;
  int N_min = 20;
  const double m = 5; // The mass of atoms. Doesn't matter in minimization.
  double dt = dt_0;
  double alpha = alpha_start;
  int N_neg = 0;
  double P;
  double max_move = 0.2;

public:
  Minimizer_FIRE_JQH(
    const int number_of_atoms,
    const int number_of_steps,
    const double force_tolerance);

  void parse_FIRE(const char** param, int num_param, int nstart);

  void compute(
    Force& force,
    Box& box,
    GPU_Vector<double>& position_per_atom,
    GPU_Vector<int>& type,
    std::vector<Group>& group,
    GPU_Vector<double>& potential_per_atom,
    GPU_Vector<double>& force_per_atom,
    GPU_Vector<double>& virial_per_atom);
    
  void compute(BaseAtoms& atoms);
};