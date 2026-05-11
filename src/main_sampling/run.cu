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
Run simulation according to the inputs in the run.in file.
------------------------------------------------------------------------------*/

#include "run.cuh"

#include "gas-metad.cuh"
#include "utilities/error.cuh"
#include "utilities/gpu_macro.cuh"
#include <chrono>
#include <cstring>

namespace {
void gpu_sampling_dump_restart(Box& box, std::vector<Group>& group, Atom& atom)
{
  FILE* fid = my_fopen("restart.xyz", "w");

  const int number_of_atoms = atom.number_of_atoms;

  atom.position_per_atom.copy_to_host(atom.cpu_position_per_atom.data());
  atom.velocity_per_atom.copy_to_host(atom.cpu_velocity_per_atom.data());

  fprintf(fid, "%d\n", number_of_atoms);

  fprintf(
    fid,
    "pbc=\"%c %c %c\" ",
    box.pbc_x ? 'T' : 'F',
    box.pbc_y ? 'T' : 'F',
    box.pbc_z ? 'T' : 'F');

  fprintf(
    fid,
    "Lattice=\"%g %g %g %g %g %g %g %g %g\" ",
    box.cpu_h[0],
    box.cpu_h[3],
    box.cpu_h[6],
    box.cpu_h[1],
    box.cpu_h[4],
    box.cpu_h[7],
    box.cpu_h[2],
    box.cpu_h[5],
    box.cpu_h[8]);

  if (group.size() == 0) {
    fprintf(fid, "Properties=species:S:1:pos:R:3:mass:R:1:vel:R:3\n");
  } else {
    fprintf(fid, "Properties=species:S:1:pos:R:3:mass:R:1:vel:R:3:group:I:%d\n", int(group.size()));
  }

  for (int n = 0; n < number_of_atoms; n++) {
    const double natural_to_A_per_fs = 1.0 / TIME_UNIT_CONVERSION;
    fprintf(
      fid,
      "%s %g %g %g %g %g %g %g ",
      atom.cpu_atom_symbol[n].c_str(),
      atom.cpu_position_per_atom[n],
      atom.cpu_position_per_atom[n + number_of_atoms],
      atom.cpu_position_per_atom[n + 2 * number_of_atoms],
      atom.cpu_mass[n],
      atom.cpu_velocity_per_atom[n] * natural_to_A_per_fs,
      atom.cpu_velocity_per_atom[n + number_of_atoms] * natural_to_A_per_fs,
      atom.cpu_velocity_per_atom[n + 2 * number_of_atoms] * natural_to_A_per_fs);

    for (int m = 0; m < int(group.size()); ++m) {
      fprintf(fid, "%d ", group[m].cpu_label[n]);
    }

    fprintf(fid, "\n");
  }

  fflush(fid);
  fclose(fid);
}
} // namespace

static __global__ void gpu_find_largest_v2(
  int N,
  int number_of_rounds,
  double* g_vx,
  double* g_vy,
  double* g_vz,
  double* g_v2_max)
{
  int tid = threadIdx.x;
  __shared__ double s_data[1024];
  s_data[tid] = 0.0;
  for (int round = 0; round < number_of_rounds; ++round) {
    int n = round * 1024 + tid;
    if (n < N) {
      double vx = g_vx[n];
      double vy = g_vy[n];
      double vz = g_vz[n];
      double v2 = vx * vx + vy * vy + vz * vz;
      if (s_data[tid] < v2) {
        s_data[tid] = v2;
      }
    }
  }
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      if (s_data[tid] < s_data[tid + offset]) {
        s_data[tid] = s_data[tid + offset];
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    g_v2_max[0] = s_data[0];
  }
}

__device__ double device_v2_max[1];

static void calculate_time_step(
  double max_distance_per_step,
  GPU_Vector<double>& velocity_per_atom,
  double initial_time_step,
  double& time_step)
{
  if (max_distance_per_step <= 0.0) {
    return;
  }
  const int N = velocity_per_atom.size() / 3;
  double* gpu_v2_max;
  CHECK(gpuGetSymbolAddress((void**)&gpu_v2_max, device_v2_max));
  gpu_find_largest_v2<<<1, 1024>>>(
    N,
    (N - 1) / 1024 + 1,
    velocity_per_atom.data(),
    velocity_per_atom.data() + N,
    velocity_per_atom.data() + N * 2,
    gpu_v2_max);
  GPU_CHECK_KERNEL
  double cpu_v2_max[1] = {0.0};
  CHECK(gpuMemcpy(cpu_v2_max, gpu_v2_max, sizeof(double), gpuMemcpyDeviceToHost));
  double cpu_v_max = sqrt(cpu_v2_max[0]);
  double time_step_min = max_distance_per_step / cpu_v_max;

  if (time_step_min < initial_time_step) {
    time_step = time_step_min;
  } else {
    time_step = initial_time_step;
  }
}

GSRun::GSRun() : GSRun("model.xyz", "run.in") {}

GSRun::GSRun(const std::string& model_filename) : GSRun(model_filename, "run.in") {}

GSRun::GSRun(const std::string& model_filename, const std::string& run_filename)
  : Run(model_filename, run_filename, false)
{
  execute_run_in();
}

void GSRun::perform_a_run()
{
  integrate.initialize(time_step, atom, box, group, thermo, number_of_steps);
  mc.initialize();
  measure.initialize(number_of_steps, time_step, integrate, group, atom, box, force);

  if (integrate.type >= 31) {
    for (int k = 0; k < integrate.number_of_beads; ++k) {
      force.compute(
        box,
        atom.position_beads[k],
        atom.type,
        group,
        atom.potential_beads[k],
        atom.force_beads[k],
        atom.virial_beads[k],
        atom.velocity_beads[k],
        atom.mass);
    }
  } else {
    force.compute(
      box,
      atom.position_per_atom,
      atom.type,
      group,
      atom.potential_per_atom,
      atom.force_per_atom,
      atom.virial_per_atom,
      atom.velocity_per_atom,
      atom.mass);
  }

  double initial_time_step = time_step;
  const auto time_begin = std::chrono::high_resolution_clock::now();

  for (int step = 0; step < number_of_steps; ++step) {
    velocity.correct_velocity(step, group, atom);

    calculate_time_step(max_distance_per_step, atom.velocity_per_atom, initial_time_step, time_step);
    global_time += time_step;

    integrate.current_step = step;
    integrate.compute1(time_step, double(step) / number_of_steps, group, box, atom, thermo);

    if (integrate.type >= 31) {
      for (int k = 0; k < integrate.number_of_beads; ++k) {
        force.compute(
          box,
          atom.position_beads[k],
          atom.type,
          group,
          atom.potential_beads[k],
          atom.force_beads[k],
          atom.virial_beads[k],
          atom.velocity_beads[k],
          atom.mass);
      }
    } else {
      force.compute(
        box,
        atom.position_per_atom,
        atom.type,
        group,
        atom.potential_per_atom,
        atom.force_per_atom,
        atom.virial_per_atom,
        atom.velocity_per_atom,
        atom.mass);
    }

    electron_stop.compute(time_step, atom);
    add_force.compute(step, group, atom);
    add_spring.compute(step, group, atom);
    add_random_force.compute(step, atom);
    add_efield.compute(step, group, atom, force);

    integrate.compute2(time_step, double(step) / number_of_steps, group, box, atom, thermo, force);

    mc.compute(step, number_of_steps, atom, box, group);

    measure.process(
      number_of_steps,
      step,
      integrate.fixed_group,
      integrate.move_group,
      global_time,
      integrate.temperature2,
      integrate,
      box,
      group,
      thermo,
      atom,
      force);

    int base = (10 <= number_of_steps) ? (number_of_steps / 10) : 1;
    if (0 == (step + 1) % base) {
      printf("    %d steps completed.\n", step + 1);
      fflush(stdout);
    }

    if (is_pathsampling) {
      bool is_match = p_gasps->process(box, atom.position_per_atom);
      if (is_match) {
        printf("[PathSampling] Reached (Meta)stable phase, computation ended.\n");
        fflush(stdout);
        gpu_sampling_dump_restart(box, group, atom);
        break;
      }
    }
    if (is_ffs) {
      bool is_end = p_gasps->process(box, atom.position_per_atom, p_gasps->config.target_stage);
      if (is_end) {
        printf("[ForwardFluxSampling] Reached NEXT/INIT phase, computation ended.\n");
        fflush(stdout);
        gpu_sampling_dump_restart(box, group, atom);
        break;
      }
    }
  }

  print_line_1();
  const auto time_finish = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double> time_used = time_finish - time_begin;

  printf("Time used for this run = %g second.\n", time_used.count());
  double run_speed = atom.number_of_atoms * (number_of_steps * 1.0 / time_used.count());
  printf("Speed of this run = %g atom*step/second.\n", run_speed);
  print_line_2();

  measure.finalize(atom, box, integrate, number_of_steps, time_step, integrate.temperature2);

  electron_stop.finalize();
  add_force.finalize();
  add_spring.finalize();
  add_random_force.finalize();
  add_efield.finalize();
  integrate.finalize();
  mc.finalize();
  velocity.finalize();
  force.finalize();
  max_distance_per_step = 0.0;
}

void GSRun::parse_one_keyword(std::vector<std::string>& tokens)
{
  int num_param = tokens.size();
  const int max_num_param = 32;
  if (num_param > max_num_param) {
    PRINT_INPUT_ERROR("The number of parameters should be less than 32.\n");
  }

  const char* param[max_num_param];
  for (int n = 0; n < num_param; ++n) {
    param[n] = tokens[n].c_str();
  }

  if (strcmp(param[0], "GASMD") == 0 || strcmp(param[0], "MetaD") == 0) {
    std::unique_ptr<TorchMetad> p_gas_metad = TorchMetad::parse_GASMD(param, num_param, atom.number_of_atoms);
    force.potentials.emplace_back(std::move(p_gas_metad));
    force.set_multiple_potentials_mode("sum");
    return;
  }

  if (strcmp(param[0], "PathSampling") == 0) {
    p_gasps = TorchMonitor::parse_GASMon(param, num_param, atom.number_of_atoms);
    is_pathsampling = true;
    is_ffs = false;
    return;
  }

  if (strcmp(param[0], "FFSampling") == 0) {
    p_gasps = TorchMonitor::parse_GASMon(param, num_param, atom.number_of_atoms);
    is_ffs = true;
    is_pathsampling = false;
    return;
  }

  Run::parse_one_keyword(tokens);
}
