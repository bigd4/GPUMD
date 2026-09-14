#include "output_frame.cuh"
namespace trajectory {
namespace {
__global__ void sum_virial(int n, const double* input, double* output) {
  __shared__ double sum[256];
  int component=blockIdx.x;
  double v=0;for(int i=threadIdx.x;i<n;i+=blockDim.x)v+=input[component*n+i];
  sum[threadIdx.x]=v;__syncthreads();
  for(int offset=128;offset;offset/=2) {if(threadIdx.x<offset)sum[threadIdx.x]+=sum[threadIdx.x+offset];__syncthreads();}
  if(threadIdx.x==0)output[component]=sum[0];
}
}
std::string thermodynamics(GPU_Vector<double>& virial, GPU_Vector<double>& thermo) {
  GPU_Vector<double> totals(6);double v[6], t[8];
  sum_virial<<<6,256>>>(virial.size()/9,virial.data(),totals.data());
  totals.copy_to_host(v);thermo.copy_to_host(t,8);
  const double matrix[9]={v[0],v[3],v[4],v[3],v[1],v[5],v[4],v[5],v[2]};
  const double stress[9]={t[2],t[5],t[6],t[5],t[3],t[7],t[6],t[7],t[4]};
  std::ostringstream s;s.imbue(std::locale::classic());s<<std::setprecision(17)<<" energy="<<t[1];
  return s.str()+tensor("virial",matrix,9)+tensor("stress",stress,9);
}
}
