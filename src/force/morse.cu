#include "morse.cuh"
#include "model/read_xyz.cuh"


// Confined Morse
static __device__ void find_force_morse(
  double md, double ma, double mz, double mw, 
  double mzz, double mk0, double& mp2, double& mf2)
{
  double term1 = 1.0 - exp(-ma * (mw/2 - mzz - mz));
  double term2 = 1.0 - exp(-ma * (mw/2 + mzz - mz));
  mp2 = md * (pow(term1, 2) - 1) + \
        md * (pow(term2, 2) - 1) + \
        mk0;
  mf2 = -1 * (2 * ma * md * exp(-ma * (mw/2 + mzz - mz)) * term2 - \
              2 * ma * md * exp(-ma * (mw/2 - mzz - mz)) * term1);
}

static __global__ void calc_morse_force(
  const int natoms,
  const double lz,
  const int itype,
  const double md,
  const double ma,
  const double mz,
  const double mw,
  const double mk0,
  const int* g_type,
  double* e0,
  const double* g_z,
  double* f0_z)
{
  int nid = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (nid < natoms){
    if (g_type[nid] != itype) return;
    double mp, mf; // morse potential, morse force
    find_force_morse(md, ma, mz, mw, g_z[nid] - lz/2, mk0, mp, mf);
    atomicAdd(&e0[nid], mp);
    atomicAdd(&f0_z[nid], mf);
  }
}

Morse::Morse()
{
}

void Morse::parse_morse(const char** param, int num_param, Force& force)
{
  p_force = &force;
  if (!is_valid_int(param[1], &itype)) {
    PRINT_INPUT_ERROR("itype should be an int.");
  }
  for (int n=2; n<num_param; n++){
    if (strcmp(param[n], "md") == 0) {
      if (!is_valid_real(param[n+1], &md)) {
        PRINT_INPUT_ERROR("md should be a real.");
      }
      n++;
    } else if (strcmp(param[n], "ma") == 0) {
      if (!is_valid_real(param[n+1], &ma)) {
        PRINT_INPUT_ERROR("ma should be a real.");
      }
      n++;
    } else if (strcmp(param[n], "mz") == 0) {
      if (!is_valid_real(param[n+1], &mz)) {
        PRINT_INPUT_ERROR("mz should be a real.");
      }
      n++;
    } else if (strcmp(param[n], "mw") == 0) {
      if (!is_valid_real(param[n+1], &mw)) {
        PRINT_INPUT_ERROR("mw should be a real.");
      }
      n++;
    } else if (strcmp(param[n], "mk0") == 0) {
      if (!is_valid_real(param[n+1], &mk0)) {
        PRINT_INPUT_ERROR("mk0 should be a real.");
      }
      n++;
    } else {
    PRINT_INPUT_ERROR(("no keyword match with: " + std::string(param[n])).data());
  }
  }
  printf("--------morse potential settings----------\n");
  printf("        type = %d\n", itype);
  printf("          md = %g\n", md);
  printf("          ma = %g\n", ma);
  printf("          mz = %g\n", mz);
  printf("          mw = %g\n", mw);
  printf("         mk0 = %g\n", mk0);
  printf("-------------------------------------\n");

}


void Morse::compute(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  natoms = type.size();
  double lz = box.thickness_z;

  calc_morse_force<<<(natoms-1)/128+1, 128>>>(
    natoms,
    lz,
    itype,
    md,
    ma,
    mz,
    mw,
    mk0,
    type.data(),
    potential_per_atom.data(),
    position_per_atom.data() + natoms * 2,
    force_per_atom.data() + natoms * 2
  );
  step++;
  
}