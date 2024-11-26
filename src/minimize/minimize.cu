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

/*----------------------------------------------------------------------------80
The driver class for minimizers.
------------------------------------------------------------------------------*/

#include "force/force.cuh"
#include "minimize.cuh"
#include "minimizer_fire.cuh"
#include "minimizer_fire_jqh.cuh"
#include "minimizer_sd.cuh"
#include "utilities/error.cuh"
#include "utilities/read_file.cuh"
#include <cstring>
#include <memory>
#include "measure/dump_position.cuh"

void Minimize::parse_minimize(
  const char** param,
  int num_param,
  Force& force,
  Box& box,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<int>& type,
  std::vector<Group>& group,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom,
  const std::vector<string>& cpu_atom_symbol)
{

  int minimizer_type = 0;
  int number_of_steps = 0;
  bool vc = false;
  int n = 4;
  vector<double> pressure = {0.0};
  double force_tolerance = 0.0;
  std::unique_ptr<Minimizer> minimizer;
  const int number_of_atoms = type.size();

  if (strcmp(param[1], "sd") == 0) {
    minimizer_type = 0;

    if (num_param != 4) {
      PRINT_INPUT_ERROR("minimize sd should have 2 parameters.");
    }

    if (!is_valid_real(param[2], &force_tolerance)) {
      PRINT_INPUT_ERROR("Force tolerance should be a number.");
    }

    if (!is_valid_int(param[3], &number_of_steps)) {
      PRINT_INPUT_ERROR("Number of steps should be an integer.");
    }
    if (number_of_steps <= 0) {
      PRINT_INPUT_ERROR("Number of steps should > 0.");
    }
  } else if (strcmp(param[1], "fire") == 0) {
    minimizer_type = 1;

    if (num_param != 4) {
      PRINT_INPUT_ERROR("minimize fire should have 2 parameters.");
    }

    if (!is_valid_real(param[2], &force_tolerance)) {
      PRINT_INPUT_ERROR("Force tolerance should be a number.");
    }

    if (!is_valid_int(param[3], &number_of_steps)) {
      PRINT_INPUT_ERROR("Number of steps should be an integer.");
    }
    if (number_of_steps <= 0) {
      PRINT_INPUT_ERROR("Number of steps should > 0.");
    }
  } else if (strcmp(param[1], "vcfire") == 0) {
    vc = true;
    minimizer_type = 1;

    if (num_param < 4) {
      PRINT_INPUT_ERROR("minimize vcfire should have at least 2 parameters: force_tol, nsteps.");
    }

    if (!is_valid_real(param[2], &force_tolerance)) {
      PRINT_INPUT_ERROR("Force tolerance should be a number.");
    }

    if (!is_valid_int(param[3], &number_of_steps)) {
      PRINT_INPUT_ERROR("Number of steps should be an integer.");
    }
    if (number_of_steps <= 0) {
      PRINT_INPUT_ERROR("Number of steps should > 0.");
    }
    if (strcmp(param[n], "p") == 0){
      if (!is_valid_real(param[n+1], &pressure[0])) {
        PRINT_INPUT_ERROR("p should be an real.");
      }
      n += 2;
    } else if (strcmp(param[n], "p3") == 0){
      pressure.resize(3);
      for (int i=0; i<3; i++){
        if (!is_valid_real(param[n+1+i], &pressure[i])) {
          PRINT_INPUT_ERROR("p3 should be 3 reals.");
        }
      }
      n += 4;
    } else if (strcmp(param[n], "p6") == 0){
      vector<double> press_in(6);
      pressure.resize(9);
      for (int i=0; i<6; i++){
        if (!is_valid_real(param[n+1+i], &press_in[i])) {
          PRINT_INPUT_ERROR("p6 should be 6 reals.");
        }
      }
      pressure[0] = press_in[0];
      pressure[4] = press_in[1];
      pressure[8] = press_in[2];
      pressure[5] = pressure[7] = press_in[3];
      pressure[2] = pressure[6] = press_in[4];
      pressure[1] = pressure[3] = press_in[5];
      n += 7;
    } else {
      PRINT_INPUT_ERROR("Invalid input for vcfire.");
    }
  }

  switch (minimizer_type) {
    case 0:
      printf("\nStart to do an energy minimization.\n");
      printf("    using the steepest descent method.\n");
      printf("    with fixed box.\n");
      printf("    with a force tolerance of %g eV/A.\n", force_tolerance);
      printf("    for maximally %d steps.\n", number_of_steps);

      minimizer.reset(new Minimizer_SD(number_of_atoms, number_of_steps, force_tolerance));

      minimizer->compute(
        force,
        box,
        position_per_atom,
        type,
        group,
        potential_per_atom,
        force_per_atom,
        virial_per_atom);

      break;
    case 1:
      printf("\nStart to do an energy minimization.\n");
      printf("    using the fast inertial relaxation engine (FIRE) method.\n");
      printf("    with fixed box.\n");
      printf("    with a force tolerance of %g eV/A.\n", force_tolerance);
      printf("    for maximally %d steps.\n", number_of_steps);

      if (vc){
        printf("variable cell is enabled.\n");
        vector<double> press={pressure};
        Atoms atoms(force, box, position_per_atom, type, group, potential_per_atom, force_per_atom, virial_per_atom);
        minimizer.reset(new Minimizer_FIRE_JQH(number_of_atoms+3, number_of_steps, force_tolerance));
        dynamic_cast<Minimizer_FIRE_JQH&>(*minimizer).parse_FIRE(param, num_param, n);
        VCWrapper& vcatoms = *new VCWrapper(atoms, press);
        vcatoms.optimize_factor = pow(atoms.get_natoms(), 1.0/4);
        printf("cell_factor = %f, optimize_factor = %f\n", vcatoms.cell_factor, vcatoms.optimize_factor);
        vcatoms.build_positions();
        // vcatoms.compute();
        // printf("    initial enthalpy = %f eV\n", vcatoms.get_energy());
        minimizer->compute(vcatoms);
        printf("    final enthalpy = %f eV\n", vcatoms.get_energy());
        box = atoms.box;
        position_per_atom = atoms.get_positions();
        potential_per_atom = atoms.get_potential_per_atom();
        force_per_atom = atoms.get_forces();
        virial_per_atom = atoms.virials;
        if (cpu_atom_symbol.size() > 0){
          FILE *fid = my_fopen("relaxed.xyz", "w");
          save_one_frame(fid, box, atoms.get_energy(), vcatoms.get_energy(), cpu_atom_symbol, position_per_atom);
          fclose(fid);
        }
      } else{
        minimizer.reset(new Minimizer_FIRE(number_of_atoms, number_of_steps, force_tolerance));

        minimizer->compute(
          force,
          box,
          position_per_atom,
          type,
          group,
          potential_per_atom,
          force_per_atom,
          virial_per_atom);
      }
      break;
    default:
      PRINT_INPUT_ERROR("Invalid minimizer.");
      break;
  }
}

// void Minimize::parse_minimize(
//   const char** param,
//   int num_param,
//   Force& force,
//   Box& box,
//   std::vector<Group>& group,
//   Atom& atom)
// {
//   parse_minimize(
//       param,
//       num_param,
//       force,
//       box,
//       atom.position_per_atom,
//       atom.type,
//       group,
//       atom.potential_per_atom,
//       atom.force_per_atom,
//       atom.virial_per_atom,
//       atom.cpu_atom_symbol);
// }