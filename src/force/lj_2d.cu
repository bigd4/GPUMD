#include "lj_2d.cuh"
#include "model/read_xyz.cuh"

// LJ93 for confined H2O from Xiaocheng Zeng
static __device__ void find_force_lj93_zeng(
  double mepsilon, double mw, double msigma, double mzz, double& mp2, double& mf2)
{
  mp2 = 4.0 * mepsilon * (
        pow(msigma / (mw/2.0 - mzz), 9) - pow(msigma / (mw/2.0 - mzz), 3)
      + pow(msigma / (mw/2.0 + mzz), 9) - pow(msigma / (mw/2.0 + mzz), 3)
    );
  mf2 = -4.0 * mepsilon * (
        9.0 * pow(msigma / (mw/2.0 - mzz), 10) - 3.0 * pow(msigma / (mw/2.0 - mzz), 4)
      - 9.0 * pow(msigma / (mw/2.0 + mzz), 10) + 3.0 * pow(msigma / (mw/2.0 + mzz), 4)
    );
}

static __global__ void calc_LJ_2d_force(
  const int natoms,
  const double lz,
  const int itype,
  const double mepsilon,
  const double mw,
  const double msigma,
  const int* g_type,
  double* e0,
  const double* g_z,
  double* f0_z)
{
  int nid = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (nid < natoms){
    if (g_type[nid] != itype) return;
    double mp, mf; // LJ_2d potential, LJ_2d force
    find_force_lj93_zeng(mepsilon, mw, msigma, g_z[nid] - lz/2, mp, mf);
    atomicAdd(&e0[nid], mp);
    atomicAdd(&f0_z[nid], mf);
  }
}

LJ_2d::LJ_2d()
{
}

void LJ_2d::parse_lj_2d(const char** param, int num_param, Force& force)
{
  p_force = &force;
  if (!is_valid_int(param[1], &itype)) {
    PRINT_INPUT_ERROR("itype should be an int.");
  }
  for (int n=2; n<num_param; n++){
    if (strcmp(param[n], "mepsilon") == 0) {
      if (!is_valid_real(param[n+1], &mepsilon)) {
        PRINT_INPUT_ERROR("mepsilon should be a real.");
      }
      n++;
    } else if (strcmp(param[n], "mw") == 0) {
      if (!is_valid_real(param[n+1], &mw)) {
        PRINT_INPUT_ERROR("mw should be a real.");
      }
      n++;
    } else if (strcmp(param[n], "msigma") == 0) {
      if (!is_valid_real(param[n+1], &msigma)) {
        PRINT_INPUT_ERROR("msigma should be a real.");
      }
      n++;
    } else {
    PRINT_INPUT_ERROR(("no keyword match with: " + std::string(param[n])).data());
  }
  }
  printf("--------LJ_2d potential settings----------\n");
  printf("           type = %d\n", itype);
  printf("       mepsilon = %g\n", mepsilon);
  printf("             mw = %g\n", mw);
  printf("         msigma = %g\n", msigma);
  printf("-------------------------------------\n");

}


void LJ_2d::compute(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  natoms = type.size();
  double lz = box.thickness_z;

  calc_LJ_2d_force<<<(natoms-1)/128+1, 128>>>(
    natoms,
    lz,
    itype,
    mepsilon,
    mw,
    msigma,
    type.data(),
    potential_per_atom.data(),
    position_per_atom.data() + natoms * 2,
    force_per_atom.data() + natoms * 2
  );
  step++;
  
}