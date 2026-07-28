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
The FIRE (fast inertial relaxation engine) minimizer
Reference: PhysRevLett 97, 170201 (2006)
           Computational Materials Science 175 (2020) 109584
------------------------------------------------------------------------------*/

#include "minimizer_fire_jqh.cuh"
#include <algorithm>
#include <numeric>
using namespace std;

namespace
{
__global__ void gpu_multiply(const int size, double a, double* b, double* c)
{
  int n = blockDim.x * blockIdx.x + threadIdx.x;
  if (n < size)
    c[n] = b[n] * a;
}

__global__ void gpu_vector_add(const int size, double* a, double* b, double* c)
{
  int n = blockDim.x * blockIdx.x + threadIdx.x;
  if (n < size)
    c[n] = a[n] + b[n];
}

__global__ void gpu_pairwise_product(const int size, double* a, double* b, double* c)
{
  int n = blockDim.x * blockIdx.x + threadIdx.x;
  if (n < size)
    c[n] = a[n] * b[n];
}

__global__ void gpu_cell_metric_copy(
  const int size,
  const int atoms_per_block,
  const int real_atoms_per_block,
  const double cell_metric_scale,
  const double* src,
  double* dst)
{
  int n = blockDim.x * blockIdx.x + threadIdx.x;
  if (n < size) {
    int block_size = atoms_per_block * 3;
    int local = n % block_size;
    dst[n] = (local >= real_atoms_per_block * 3) ? src[n] / cell_metric_scale : src[n];
  }
}

__global__ void gpu_imagewise_reduce(
  const int image_size,
  const double* velocity,
  const double* force,
  double* power,
  double* velocity_square,
  double* force_square)
{
  const int image = blockIdx.x;
  const int offset = image * image_size;
  double local_power = 0.0;
  double local_velocity_square = 0.0;
  double local_force_square = 0.0;
  for (int component = threadIdx.x; component < image_size; component += blockDim.x) {
    const double velocity_value = velocity[offset + component];
    const double force_value = force[offset + component];
    local_power += velocity_value * force_value;
    local_velocity_square += velocity_value * velocity_value;
    local_force_square += force_value * force_value;
  }

  __shared__ double block_power[256];
  __shared__ double block_velocity_square[256];
  __shared__ double block_force_square[256];
  block_power[threadIdx.x] = local_power;
  block_velocity_square[threadIdx.x] = local_velocity_square;
  block_force_square[threadIdx.x] = local_force_square;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      block_power[threadIdx.x] += block_power[threadIdx.x + stride];
      block_velocity_square[threadIdx.x] +=
        block_velocity_square[threadIdx.x + stride];
      block_force_square[threadIdx.x] += block_force_square[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    power[image] = block_power[0];
    velocity_square[image] = block_velocity_square[0];
    force_square[image] = block_force_square[0];
  }
}

__global__ void gpu_imagewise_reset(
  const int size,
  const int image_size,
  const double* image_dt,
  const int* reset,
  double* position,
  double* velocity)
{
  const int component = blockDim.x * blockIdx.x + threadIdx.x;
  if (component >= size) return;
  const int image = component / image_size;
  if (reset[image] == 1) {
    position[component] -= 0.5 * image_dt[image] * velocity[component];
    velocity[component] = 0.0;
  } else if (reset[image] == 2) {
    velocity[component] = 0.0;
  }
}

__global__ void gpu_imagewise_integrate(
  const int size,
  const int image_size,
  const double inverse_mass,
  const double* image_dt,
  const double* one_minus_alpha,
  const double* mixing_scale,
  const double* force,
  double* velocity,
  double* displacement)
{
  const int component = blockDim.x * blockIdx.x + threadIdx.x;
  if (component >= size) return;
  const int image = component / image_size;
  double velocity_value =
    velocity[component] + image_dt[image] * inverse_mass * force[component];
  velocity_value =
    one_minus_alpha[image] * velocity_value +
    mixing_scale[image] * force[component];
  velocity[component] = velocity_value;
  displacement[component] = image_dt[image] * velocity_value;
}

__global__ void gpu_imagewise_metric_max(
  const int image_size,
  const int real_size,
  const double cell_metric_scale,
  const double* displacement,
  double* displacement_max)
{
  const int image = blockIdx.x;
  const int offset = image * image_size;
  double local_max = 0.0;
  for (int component = threadIdx.x; component < image_size; component += blockDim.x) {
    double value = abs(displacement[offset + component]);
    if (component >= real_size) value /= cell_metric_scale;
    local_max = max(local_max, value);
  }

  __shared__ double block_max[256];
  block_max[threadIdx.x] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      block_max[threadIdx.x] =
        max(block_max[threadIdx.x], block_max[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) displacement_max[image] = block_max[0];
}

__global__ void gpu_imagewise_apply_move(
  const int size,
  const int image_size,
  const double max_move,
  const double* displacement_max,
  const double* displacement,
  double* position)
{
  const int component = blockDim.x * blockIdx.x + threadIdx.x;
  if (component >= size) return;
  const int image = component / image_size;
  const double scale =
    displacement_max[image] > max_move ?
      max_move / displacement_max[image] :
      1.0;
  position[component] += scale * displacement[component];
}

void pairwise_product(GPU_Vector<double>& a, GPU_Vector<double>& b, GPU_Vector<double>& c)
{
  int size = a.size();
  gpu_pairwise_product<<<(size - 1) / 128 + 1, 128>>>(size, a.data(), b.data(), c.data());
}

__global__ void gpu_sum(const int size, double* a, double* result)
{
  int number_of_patches = (size - 1) / 1024 + 1;
  int tid = threadIdx.x;
  int n, patch;
  __shared__ double data[1024];
  data[tid] = 0.0;
  for (patch = 0; patch < number_of_patches; ++patch) {
    n = tid + patch * 1024;
    if (n < size)
      data[tid] += a[n];
  }
  __syncthreads();
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      data[tid] += data[tid + offset];
    }
    __syncthreads();
  }
  if (tid == 0)
    *result = data[0];
}

double sum(GPU_Vector<double>& a)
{
  double ret;
  GPU_Vector<double> result(1);
  gpu_sum<<<1, 1024>>>(a.size(), a.data(), result.data());
  result.copy_to_host(&ret);
  return ret;
}

double dot(GPU_Vector<double>& a, GPU_Vector<double>& b)
{
  GPU_Vector<double> temp(a.size());
  pairwise_product(a, b, temp);
  return sum(temp);
}

void scalar_multiply(const double& a, GPU_Vector<double>& b, GPU_Vector<double>& c)
{
  int size = b.size();
  gpu_multiply<<<(size - 1) / 128 + 1, 128>>>(size, a, b.data(), c.data());
}

void vector_add(GPU_Vector<double>& a, GPU_Vector<double>& b, GPU_Vector<double>& c)
{
  int size = a.size();
  gpu_vector_add<<<(size - 1) / 128 + 1, 128>>>(size, a.data(), b.data(), c.data());
}

cublasHandle_t handle;
double max_abs(int size, double* vec)
{
  int index1;
  double result;
  cublasIdamax(handle, size, vec, 1, &index1);
  int index0 = index1 - 1;
  cudaMemcpy(&result, vec + index0, sizeof(double), cudaMemcpyDeviceToHost);
  return abs(result);
}

double metric_max_abs(BaseAtoms& atoms, GPU_Vector<double>& vec, GPU_Vector<double>& metric_vec, double cell_metric_scale)
{
  const int size = vec.size();
  if (!atoms.has_cell_degrees_of_freedom()) {
    return max_abs(size, vec.data());
  }
  const int atoms_per_block = atoms.get_atoms_per_block();
  const int real_atoms_per_block = atoms.get_real_atom_count_per_block();
  if (atoms_per_block <= real_atoms_per_block || real_atoms_per_block <= 0 || cell_metric_scale == 1.0) {
    return max_abs(size, vec.data());
  }
  metric_vec.resize(size);
  gpu_cell_metric_copy<<<(size - 1) / 128 + 1, 128>>>(
    size,
    atoms_per_block,
    real_atoms_per_block,
    cell_metric_scale,
    vec.data(),
    metric_vec.data());
  GPU_CHECK_KERNEL;
  return max_abs(size, metric_vec.data());
}
} // namespace

Minimizer_FIRE_JQH::Minimizer_FIRE_JQH(
  const int number_of_atoms, const int number_of_steps, const double force_tolerance)
  : Minimizer(number_of_atoms, number_of_steps, force_tolerance)
{
  cublasCreate(&handle);
}

void Minimizer_FIRE_JQH::parse_FIRE(const char** param, int num_param, int nstart, bool printflag0)
{
  printflag = printflag0;
  for (int n=nstart; n<num_param; n++){
    if (strcmp(param[n], "max_move") == 0){
      require_option_values(param, num_param, n, 1, "vcfire");
      if (!is_valid_real(param[n+1], &max_move)) {
        PRINT_INPUT_ERROR("max_move should be a number.");
      }
      n++;
    } else if (strcmp(param[n], "cell_metric_scale") == 0){
      require_option_values(param, num_param, n, 1, "vcfire");
      if (!is_valid_real(param[n+1], &cell_metric_scale)) {
        PRINT_INPUT_ERROR("cell_metric_scale should be a number.");
      }
      if (cell_metric_scale <= 0.0) {
        PRINT_INPUT_ERROR("cell_metric_scale should > 0.");
      }
      n++;
    } else if (strcmp(param[n], "dt_max") == 0){
      require_option_values(param, num_param, n, 1, "vcfire");
      double tmp_dt_max;
      if (!is_valid_real(param[n+1], &tmp_dt_max)) {
        PRINT_INPUT_ERROR("dt_max should be a number.");
      }
      dt_max = tmp_dt_max / TIME_UNIT_CONVERSION;
      n++;
    } else if (strcmp(param[n], "dt_min") == 0){
      require_option_values(param, num_param, n, 1, "vcfire");
      double tmp_dt_min;
      if (!is_valid_real(param[n+1], &tmp_dt_min)) {
        PRINT_INPUT_ERROR("dt_min should be a number.");
      }
      dt_min = tmp_dt_min / TIME_UNIT_CONVERSION;
      n++;
    } else if (strcmp(param[n], "dt_0") == 0){
      require_option_values(param, num_param, n, 1, "vcfire");
      double tmp_dt_0;
      if (!is_valid_real(param[n+1], &tmp_dt_0)) {
        PRINT_INPUT_ERROR("dt_0 should be a number.");
      }
      dt_0 = tmp_dt_0 / TIME_UNIT_CONVERSION;
      dt = dt_0;
      n++;
    } else if (strcmp(param[n], "f_inc") == 0){
      require_option_values(param, num_param, n, 1, "vcfire");
      if (!is_valid_real(param[n+1], &f_inc)) {
        PRINT_INPUT_ERROR("f_inc should be a number.");
      }
      n++;
    } else if (strcmp(param[n], "alpha_start") == 0){
      require_option_values(param, num_param, n, 1, "vcfire");
      if (!is_valid_real(param[n+1], &alpha_start)) {
        PRINT_INPUT_ERROR("alpha_start should be a number.");
      }
      n++;
    } else if (strcmp(param[n], "f_alpha") == 0){
      require_option_values(param, num_param, n, 1, "vcfire");
      if (!is_valid_real(param[n+1], &f_alpha)) {
        PRINT_INPUT_ERROR("f_alpha should be a number.");
      }
      n++;
    } else if (strcmp(param[n], "alpha_min") == 0){
      require_option_values(param, num_param, n, 1, "vcfire");
      if (!is_valid_real(param[n+1], &alpha_min)) {
        PRINT_INPUT_ERROR("alpha_min should be a number.");
      }
      if (alpha_min < 0.0) {
        PRINT_INPUT_ERROR("alpha_min should be >= 0.");
      }
      n++;
    } else if (strcmp(param[n], "N_min") == 0){
      require_option_values(param, num_param, n, 1, "vcfire");
      if (!is_valid_int(param[n+1], &N_min)) {
        PRINT_INPUT_ERROR("N_min should be an int.");
      }
      n++;
    } else if (strcmp(param[n], "min_alignment_cosine") == 0){
      require_option_values(param, num_param, n, 1, "vcfire");
      if (!is_valid_real(param[n+1], &min_alignment_cosine)) {
        PRINT_INPUT_ERROR("min_alignment_cosine should be a number.");
      }
      if (min_alignment_cosine < 0.0 || min_alignment_cosine >= 1.0) {
        PRINT_INPUT_ERROR("min_alignment_cosine should be >= 0 and < 1.");
      }
      n++;
    } else if (strcmp(param[n], "imagewise") == 0){
      imagewise = true;
    } else if (strcmp(param[n], "rotation_free") == 0){
      ;
    } else {
    string text="Invalid option for vcfire: ";
    text += param[n];
    PRINT_INPUT_ERROR(text.data());
    }
  }
  if (min_alignment_cosine > 0.0 && !imagewise) {
    PRINT_INPUT_ERROR("min_alignment_cosine requires imagewise FIRE.");
  }
  if (alpha_min > alpha_start) {
    PRINT_INPUT_ERROR("alpha_min should be <= alpha_start.");
  }
  if (printflag){
    print_para();
  }
}

void Minimizer_FIRE_JQH::set_cell_metric_scale(double scale)
{
  if (scale <= 0.0) {
    PRINT_INPUT_ERROR("cell_metric_scale should > 0.");
  }
  cell_metric_scale = scale;
}

void Minimizer_FIRE_JQH::print_para(){
  printf("----------vcfire settings---------------\n");
  printf("%12s = %g\n", "max_move", max_move);
  printf("%12s = %g\n", "cell_metric_scale", cell_metric_scale);
  printf("%12s = %g\n", "dt_max", dt_max * TIME_UNIT_CONVERSION);
  printf("%12s = %g\n", "dt_min", dt_min * TIME_UNIT_CONVERSION);
  printf("%12s = %g\n", "dt_0", dt_0 * TIME_UNIT_CONVERSION);
  printf("%12s = %g\n", "f_inc", f_inc);
  printf("%12s = %g\n", "alpha_start", alpha_start);
  printf("%12s = %g\n", "f_alpha", f_alpha);
  printf("%12s = %g\n", "alpha_min", alpha_min);
  printf("%12s = %d\n", "N_min", N_min);
  printf("%12s = %s\n", "imagewise", imagewise ? "true" : "false");
  if (imagewise) {
    printf("%12s = %g\n", "min_alignment_cosine", min_alignment_cosine);
  }
  printf("----------------------------------------\n");
}

void Minimizer_FIRE_JQH::compute(
  Force& force,
  Box& box,
  Atom& atom,
  GPU_Vector<double>& position_per_atom,
  std::vector<Group>& group)
{
  if (imagewise) {
    PRINT_INPUT_ERROR("imagewise FIRE requires the BaseAtoms block interface.");
  }
  double next_dt;
  const int size = number_of_atoms_ * 3;
  int base = (number_of_steps_ >= 10) ? (number_of_steps_ / 10) : 1;
  // create a velocity vector in GPU
  GPU_Vector<double> v(size, 0);
  GPU_Vector<double> temp1(size);
  GPU_Vector<double> temp2(size);

  printf("\nEnergy minimization started.\n");

  for (int step = 0; step < number_of_steps_; ++step) {
    force.compute(
      box, position_per_atom, atom.type, group, atom.potential_per_atom, atom.force_per_atom, atom.virial_per_atom);
    calculate_force_square_max(atom.force_per_atom);
    const double force_max = sqrt(cpu_force_square_max_[0]);
    calculate_total_potential(atom.potential_per_atom);

    if (step % base == 0 || force_max < force_tolerance_) {
      printf(
        "    step %d: total_potential = %.10f eV, f_max = %.10f eV/A.\n",
        step,
        cpu_total_potential_[0],
        force_max);
      if (force_max < force_tolerance_)
        break;
    }

    P = dot(v, atom.force_per_atom);

    if (P > 0) {
      if (N_neg > N_min) {
        next_dt = dt * f_inc;
        if (next_dt < dt_max)
          dt = next_dt;
        alpha = max(alpha * f_alpha, alpha_min);
      }
      N_neg++;
    } else {
      next_dt = dt * f_dec;
      dt = max(next_dt, dt_min);
      alpha = alpha_start;
      // move position back
      scalar_multiply(-0.5 * dt, v, temp1);
      vector_add(position_per_atom, temp1, position_per_atom);
      v.fill(0);
      N_neg = 0;
    }

    // md step
    // implicit Euler integration
    double F_modulus = sqrt(dot(atom.force_per_atom, atom.force_per_atom));
    double v_modulus = sqrt(dot(v, v));
    // dv = F/m*dt
    scalar_multiply(dt / m, atom.force_per_atom, temp2);
    vector_add(v, temp2, v);
    scalar_multiply(1 - alpha, v, temp1);
    scalar_multiply(alpha * v_modulus / F_modulus, atom.force_per_atom, temp2);
    vector_add(temp1, temp2, v);
    // dx = v*dt
    scalar_multiply(dt, v, temp1);
    vector_add(position_per_atom, temp1, position_per_atom);
  }

  printf("Energy minimization finished.\n");
}

void Minimizer_FIRE_JQH::compute(BaseAtoms& atoms)
{
  if (imagewise) {
    compute_imagewise(atoms);
    return;
  }
  if (printflag) printf("---------------minimizer jqh---------------\n");
  double next_dt;
  const int size = number_of_atoms_ * 3;
  // printf("size %d, natoms %d\n", size, atoms.natoms);
  // BaseAtoms* p_atoms;
  // p_atoms = &atoms;
  int base = (number_of_steps_ >= 10) ? (number_of_steps_ / 10) : 1;
  // create a velocity vector in GPU
  GPU_Vector<double> v(size, 0);
  GPU_Vector<double> temp1(size);
  GPU_Vector<double> temp2(size);
  GPU_Vector<double> metric_temp(size);

  // GPU_Vector<double>* p_pos;
  // p_pos = &atoms.get_positions();
  // atoms.get_positions();
  // Box& box = atoms.box;
  GPU_Vector<double>& position_per_atom = atoms.get_positions();
  GPU_Vector<double>& potential_per_atom = atoms.get_potential_per_atom();
  GPU_Vector<double>& force_per_atom = atoms.get_forces();
  
  if (printflag) printf("minimizer size of positions %d\n", int(position_per_atom.size()));

  if (printflag) printf("\nEnergy minimization started.\n");
  // double h_temp1[6];

  for (int step = 0; step < number_of_steps_; ++step) {
    atoms.compute();
    // print_gpu(force_per_atom, "f");
    // print_gpu(position_per_atom, "r");
    // print_gpu(force_per_atom, "minimizer forces");
    // atoms.p_force->compute(
    //   box, position_per_atom, type, group, potential_per_atom, force_per_atom, virial_per_atom);
    const double force_max = metric_max_abs(atoms, force_per_atom, metric_temp, cell_metric_scale);
    const bool stop_after_force_max = atoms.update_minimizer_force_max(force_max);
    calculate_total_potential(potential_per_atom);

    if (step % base == 0 || force_max < force_tolerance_ || stop_after_force_max) {
      if (printflag) printf(
        "    step %d: total_energy = %.10f eV, f_max = %.10f eV/A.\n",
        step,
        atoms.get_energy(),
        force_max);
      fflush(stdout);
      if (force_max < force_tolerance_ || stop_after_force_max)
        break;
    }

    P = dot(v, force_per_atom);
    bool fire_reset = false;

    if (P > 0) {
      if (N_neg > N_min) {
        next_dt = dt * f_inc;
        if (next_dt < dt_max)
          dt = next_dt;
        alpha = max(alpha * f_alpha, alpha_min);
      }
      N_neg++;
    } else {
      fire_reset = true;
      next_dt = dt * f_dec;
      dt = max(next_dt, dt_min);
      alpha = alpha_start;
      // move position back
      scalar_multiply(-0.5 * dt, v, temp1);
      vector_add(position_per_atom, temp1, position_per_atom);
      v.fill(0);
      N_neg = 0;
    }

    atoms.report_minimizer_state(
      dt * TIME_UNIT_CONVERSION, P, alpha, N_neg, fire_reset);

    // md step
    // implicit Euler integration
    double F_modulus = sqrt(dot(force_per_atom, force_per_atom));
    double v_modulus = sqrt(dot(v, v));
    // dv = F/m*dt
    scalar_multiply(dt / m, force_per_atom, temp2); // temp2 = dv
    vector_add(v, temp2, v);
    scalar_multiply(1 - alpha, v, temp1);
    scalar_multiply(alpha * v_modulus / F_modulus, force_per_atom, temp2);
    vector_add(temp1, temp2, v);
    // dx = v*dt
    scalar_multiply(dt, v, temp1);  // temp1 = dr
    double dr_max = metric_max_abs(atoms, temp1, metric_temp, cell_metric_scale);
    if (dr_max > max_move) scalar_multiply(max_move/dr_max, temp1, temp1);
    vector_add(position_per_atom, temp1, position_per_atom);
    GPU_CHECK_KERNEL;

    // print_gpu(position_per_atom, "r2"); 
    // printf("sizeof minimizer pos %d\n", position_per_atom.size());
    // print_gpu(position_per_atom, "minimizer pos");
  }

  if (printflag) printf("Energy minimization finished.\n");
}

void Minimizer_FIRE_JQH::compute_imagewise(BaseAtoms& atoms)
{
  if (printflag) printf("----------imagewise minimizer jqh----------\n");
  const int size = number_of_atoms_ * 3;
  const int atoms_per_image = atoms.get_atoms_per_block();
  const int real_atoms_per_image = atoms.get_real_atom_count_per_block();
  const int image_size = atoms_per_image * 3;
  const int real_size = real_atoms_per_image * 3;
  if (
    atoms_per_image <= 0 ||
    real_atoms_per_image <= 0 ||
    real_atoms_per_image > atoms_per_image ||
    size % image_size != 0) {
    PRINT_INPUT_ERROR("Invalid block layout for imagewise FIRE.");
  }
  const int number_of_images = size / image_size;
  if (number_of_images <= 0) {
    PRINT_INPUT_ERROR("imagewise FIRE requires at least one movable image.");
  }
  if (printflag) {
    printf(
      "Imagewise FIRE: %d blocks, %d degrees of freedom per block.\n",
      number_of_images,
      image_size);
  }

  const int base = (number_of_steps_ >= 10) ? (number_of_steps_ / 10) : 1;
  GPU_Vector<double> velocity(size, 0.0);
  GPU_Vector<double> displacement(size);
  GPU_Vector<double> metric_temp(size);
  GPU_Vector<double> gpu_power(number_of_images);
  GPU_Vector<double> gpu_velocity_square(number_of_images);
  GPU_Vector<double> gpu_force_square(number_of_images);
  GPU_Vector<double> gpu_dt(number_of_images);
  GPU_Vector<double> gpu_one_minus_alpha(number_of_images);
  GPU_Vector<double> gpu_mixing_scale(number_of_images);
  GPU_Vector<double> gpu_displacement_max(number_of_images);
  GPU_Vector<int> gpu_reset(number_of_images);

  vector<double> image_dt(number_of_images, dt);
  vector<double> image_alpha(number_of_images, alpha_start);
  vector<double> image_power(number_of_images);
  vector<double> velocity_square(number_of_images);
  vector<double> force_square(number_of_images);
  vector<double> one_minus_alpha(number_of_images);
  vector<double> mixing_scale(number_of_images);
  vector<double> dt_report(number_of_images);
  vector<int> n_positive(number_of_images, 0);
  vector<int> reset(number_of_images, 0);

  GPU_Vector<double>& position_per_atom = atoms.get_positions();
  GPU_Vector<double>& potential_per_atom = atoms.get_potential_per_atom();
  GPU_Vector<double>& force_per_atom = atoms.get_forces();
  if (
    position_per_atom.size() != size ||
    force_per_atom.size() != size) {
    PRINT_INPUT_ERROR("Vector size does not match imagewise FIRE layout.");
  }

  if (printflag) printf("\nEnergy minimization started.\n");
  for (int step = 0; step < number_of_steps_; ++step) {
    atoms.compute();
    const double force_max =
      metric_max_abs(atoms, force_per_atom, metric_temp, cell_metric_scale);
    const bool stop_after_force_max = atoms.update_minimizer_force_max(force_max);
    calculate_total_potential(potential_per_atom);

    if (step % base == 0 || force_max < force_tolerance_ || stop_after_force_max) {
      if (printflag) printf(
        "    step %d: total_energy = %.10f eV, f_max = %.10f eV/A.\n",
        step,
        atoms.get_energy(),
        force_max);
      fflush(stdout);
      if (force_max < force_tolerance_ || stop_after_force_max) break;
    }

    gpu_imagewise_reduce<<<number_of_images, 256>>>(
      image_size,
      velocity.data(),
      force_per_atom.data(),
      gpu_power.data(),
      gpu_velocity_square.data(),
      gpu_force_square.data());
    GPU_CHECK_KERNEL;
    gpu_power.copy_to_host(image_power.data());
    gpu_velocity_square.copy_to_host(velocity_square.data());
    gpu_force_square.copy_to_host(force_square.data());

    for (int image = 0; image < number_of_images; ++image) {
      reset[image] = 0;
      if (force_square[image] <= 1.0e-30) {
        reset[image] = 2;
        image_power[image] = 0.0;
        one_minus_alpha[image] = 1.0 - image_alpha[image];
        mixing_scale[image] = 0.0;
        continue;
      }

      const double effective_power =
        image_power[image] -
        min_alignment_cosine *
          sqrt(velocity_square[image] * force_square[image]);
      if (effective_power > 0.0) {
        if (n_positive[image] > N_min) {
          const double next_dt = image_dt[image] * f_inc;
          if (next_dt < dt_max) image_dt[image] = next_dt;
          image_alpha[image] =
            max(image_alpha[image] * f_alpha, alpha_min);
        }
        n_positive[image]++;
      } else {
        image_dt[image] = max(image_dt[image] * f_dec, dt_min);
        image_alpha[image] = alpha_start;
        n_positive[image] = 0;
        velocity_square[image] = 0.0;
        reset[image] = 1;
      }

      one_minus_alpha[image] = 1.0 - image_alpha[image];
      mixing_scale[image] =
        image_alpha[image] *
        sqrt(velocity_square[image] / force_square[image]);
    }

    for (int image = 0; image < number_of_images; ++image) {
      dt_report[image] = image_dt[image] * TIME_UNIT_CONVERSION;
    }
    atoms.report_imagewise_minimizer_state(
      dt_report, image_power, image_alpha, n_positive, reset);
    atoms.report_minimizer_state(
      *max_element(dt_report.begin(), dt_report.end()),
      accumulate(image_power.begin(), image_power.end(), 0.0),
      *min_element(image_alpha.begin(), image_alpha.end()),
      *max_element(n_positive.begin(), n_positive.end()),
      any_of(reset.begin(), reset.end(), [](int value) { return value == 1; }));

    gpu_dt.copy_from_host(image_dt.data());
    gpu_one_minus_alpha.copy_from_host(one_minus_alpha.data());
    gpu_mixing_scale.copy_from_host(mixing_scale.data());
    gpu_reset.copy_from_host(reset.data());
    gpu_imagewise_reset<<<(size - 1) / 256 + 1, 256>>>(
      size,
      image_size,
      gpu_dt.data(),
      gpu_reset.data(),
      position_per_atom.data(),
      velocity.data());
    gpu_imagewise_integrate<<<(size - 1) / 256 + 1, 256>>>(
      size,
      image_size,
      1.0 / m,
      gpu_dt.data(),
      gpu_one_minus_alpha.data(),
      gpu_mixing_scale.data(),
      force_per_atom.data(),
      velocity.data(),
      displacement.data());
    gpu_imagewise_metric_max<<<number_of_images, 256>>>(
      image_size,
      real_size,
      cell_metric_scale,
      displacement.data(),
      gpu_displacement_max.data());
    gpu_imagewise_apply_move<<<(size - 1) / 256 + 1, 256>>>(
      size,
      image_size,
      max_move,
      gpu_displacement_max.data(),
      displacement.data(),
      position_per_atom.data());
    GPU_CHECK_KERNEL;
  }

  if (printflag) printf("Energy minimization finished.\n");
}
