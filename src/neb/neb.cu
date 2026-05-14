#include "neb.cuh"
#include <thrust/sort.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <numeric>
using namespace std;

void print_mem(const char* tag) {
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    printf("%s: used = %.2f MB\n", tag,
           (total - free) / 1024.0 / 1024.0);
}

namespace
{
  cublasHandle_t handle;
  cusolverDnHandle_t cusolverH;

  __global__ void gpu_multiply(double* result, double a, double* b, const int size)
  {
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < size)
      result[n] = b[n] * a;
  }

  __global__ void gpu_vector_add(double* result, double* a, double* b, const int size,
                                double alpha=1.0, double beta=1.0, double c=0.0)
  {
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < size)
      result[n] = alpha * a[n] + beta * b[n] + c;
  }

  __global__ void gpu_vector_add_scalar(double* result, double* a, double alpha, const int size)
  {
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < size)
      result[n] = a[n] + alpha;
  }

  void vector_add(GPU_Vector<double>& result, GPU_Vector<double>& a, GPU_Vector<double>& b,
                double alpha=1.0, double beta=1.0, double c=0.0)
  {
    int size = a.size();
    gpu_vector_add<<<(size - 1) / 128 + 1, 128>>>
      (result.data(), a.data(), b.data(), size, alpha, beta, c);
  }

  // vec result = vec a + scalar alpha
  void __attribute__((unused)) vector_add_scalar(
    GPU_Vector<double>& result, GPU_Vector<double>& a, double& alpha)
  {
    int size = a.size();
    gpu_vector_add_scalar<<<(size - 1) / 128 + 1, 128>>>
      (result.data(), a.data(), alpha, size);
  }

  __global__ void gpu_vector_substract(double* result, const int size, double* a, double* b)
  {
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < size)
      result[n] = a[n] - b[n];
  }

  // __global__ void gpu_vdot(
  //   double* result, const int nl,
  //   double* a1, double* a2, double* a3,
  //   double* b1, double* b2, double* b3, double alpha=1.0)
  // {
  //   int n = blockDim.x * blockIdx.x + threadIdx.x;
  //   if (n < nl) result[n] = alpha * (a1[n]*b1[n] + a2[n]*b2[n] + a3[n]*b3[n]);
  // }


  void vector_substract(GPU_Vector<double>& result, GPU_Vector<double>& a, GPU_Vector<double>& b, int size=0){
    if (size==0) size = a.size();
    gpu_vector_substract<<<(size*3 -1)/128 + 1,128>>>(result.data(), size, a.data(), b.data());
  }

  __global__ void gpu_pairwise_product(double* c, double* a, double* b, const int size, double alpha=1.0)
  {
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < size)
      c[n] = alpha * a[n] * b[n];
  }

  void pairwise_product(GPU_Vector<double>& a, GPU_Vector<double>& b, GPU_Vector<double>& c)
  {
    int size = a.size();
    gpu_pairwise_product<<<(size - 1) / 128 + 1, 128>>>(c.data(), a.data(), b.data(), size);
  }

  // __global__ void symmetrize_3x3(double* dst, double* src)
  // {
  //   int n = blockDim.x * blockIdx.x + threadIdx.x;
  //   if (n<9){
  //     if (n%4 == 0){
  //       dst[n] = src[n];
  //     }
  //     else if (n < 4){
  //       dst[n] = 
  //     }
  //   }
  // }
  // void n_nx3_multiply(GPU_Vector<double>& result, GPU_Vector<double>& a, GPU_Vector<double>& b,
  //                     int nl, double alpha=1.0)
  // {
  //   for (int i=0; i<3;i++) {
  //   gpu_pairwise_product<<<(nl - 1) / 128 + 1, 128>>>(
  //     result.data() + i*nl, a.data(), b.data() + i*nl, nl, alpha);
  //   }
  // }

  // void n_nx3_multiply(double* result, double* a, double* b,
  //                   int nl, double alpha=1.0)
  // {
  //   for (int i=0; i<3;i++) {
  //   gpu_pairwise_product<<<(nl - 1) / 128 + 1, 128>>>(result + i*nl, a, b + i*nl, nl, alpha);
  //   }
  // }
  void gpu_matmul(double* mA, double* mB, double* mC,
    int M, int N, int K, int transa=CUBLAS_OP_N, int transb=CUBLAS_OP_N,
    double alpha=1.0, double beta=0.0)
  {
    int lda = (transa != CUBLAS_OP_T)? M: K;
    int ldb = (transb != CUBLAS_OP_T)? K: N;
    // printf("lda: %d, ldb: %d\n",lda, ldb);
    cublasDgemm(handle, cublasOperation_t(transa), cublasOperation_t(transb),
      M, N, K, &alpha, mA, lda, mB, ldb, &beta, mC, M);
    // printf("cublas error code: %d\n", stat);
  }

  void get_3x3_inverse(double* m, double* m_inv)
  {
    double det;
      m_inv[0] = m[4] * m[8] - m[5] * m[7];
      m_inv[1] = m[2] * m[7] - m[1] * m[8];
      m_inv[2] = m[1] * m[5] - m[2] * m[4];
      m_inv[3] = m[5] * m[6] - m[3] * m[8];
      m_inv[4] = m[0] * m[8] - m[2] * m[6];
      m_inv[5] = m[2] * m[3] - m[0] * m[5];
      m_inv[6] = m[3] * m[7] - m[4] * m[6];
      m_inv[7] = m[1] * m[6] - m[0] * m[7];
      m_inv[8] = m[0] * m[4] - m[1] * m[3];
      det = m[0] * (m[4] * m[8] - m[5] * m[7]) +
            m[1] * (m[5] * m[6] - m[3] * m[8]) +
            m[2] * (m[3] * m[7] - m[4] * m[6]);
      for (int n = 0; n < 9; n++) {
        m_inv[n] /= det;
      }
  }


  void get_svd(double* A, double* S, double* U, double* VT, int m, int n)
  {
      if (A == nullptr || S == nullptr || U == nullptr || VT == nullptr) {
        PRINT_INPUT_ERROR("get_svd: A/S/U/VT must be preallocated and non-null.");
      }
      if (m <= 0 || n <= 0) {
        PRINT_INPUT_ERROR("get_svd: m and n must be positive.");
      }

      const int lda = m;
      const int ldu = m;  // jobu='A': U is m x m
      const int ldvt = n; // jobvt='A': VT is n x n
      int* devInfo = nullptr;
      double* Work = nullptr;
      int lwork = 0;

      auto cleanup = [&]() {
        if (Work != nullptr) {
          CHECK(cudaFree(Work));
        }
        if (devInfo != nullptr) {
          CHECK(cudaFree(devInfo));
        }
      };

      cusolverStatus_t status = cusolverDnDgesvd_bufferSize(cusolverH, m, n, &lwork);
      if (status != CUSOLVER_STATUS_SUCCESS) {
        cleanup();
        fprintf(stderr, "cuSOLVER Error: cusolverDnDgesvd_bufferSize failed, status=%d\n", int(status));
        exit(1);
      }

      CHECK(cudaMalloc(reinterpret_cast<void**>(&Work), sizeof(double) * lwork));
      CHECK(cudaMalloc(reinterpret_cast<void**>(&devInfo), sizeof(int)));

      signed char jobu = 'A';
      signed char jobvt = 'A';
      status = cusolverDnDgesvd(
          cusolverH,
          jobu,
          jobvt,
          m,
          n,
          A,
          lda,
          S,
          U,
          ldu,
          VT,
          ldvt,
          Work,
          lwork,
          nullptr,
          devInfo);
      if (status != CUSOLVER_STATUS_SUCCESS) {
        cleanup();
        fprintf(stderr, "cuSOLVER Error: cusolverDnDgesvd failed, status=%d\n", int(status));
        exit(1);
      }

      int h_info = 0;
      CHECK(cudaMemcpy(&h_info, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
      if (h_info < 0) {
        cleanup();
        fprintf(stderr, "cuSOLVER Error: get_svd got illegal argument at position %d\n", -h_info);
        exit(1);
      }
      if (h_info > 0) {
        cleanup();
        fprintf(stderr, "cuSOLVER Error: get_svd did not converge, info=%d\n", h_info);
        exit(1);
      }

      cleanup();
      GPU_CHECK_KERNEL
  }

  // Cholesky factorization by SVD:
  // 1) use SVD to validate positive-semidefinite-ness;
  // 2) build the strict lower-triangular Cholesky factor L by standard recursion.
  // Input A and output L are both n x n column-major matrices on device/managed memory.
  void __attribute__((unused)) get_cholesky(double* A, double* L, int n)
  {
    if (A == nullptr || L == nullptr) {
      PRINT_INPUT_ERROR("get_cholesky: A and L must be preallocated and non-null.");
    }
    if (n <= 0) {
      PRINT_INPUT_ERROR("get_cholesky: n must be positive.");
    }

    double* A_work = nullptr;
    double* S = nullptr;
    double* U = nullptr;
    double* VT = nullptr;

    auto cleanup = [&]() {
      if (A_work != nullptr) {
        CHECK(cudaFree(A_work));
      }
      if (S != nullptr) {
        CHECK(cudaFree(S));
      }
      if (U != nullptr) {
        CHECK(cudaFree(U));
      }
      if (VT != nullptr) {
        CHECK(cudaFree(VT));
      }
    };

    CHECK(cudaMalloc(reinterpret_cast<void**>(&A_work), sizeof(double) * n * n));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&S), sizeof(double) * n));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&U), sizeof(double) * n * n));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&VT), sizeof(double) * n * n));
    CHECK(cudaMemcpy(A_work, A, sizeof(double) * n * n, cudaMemcpyDefault));

    get_svd(A_work, S, U, VT, n, n);

    std::vector<double> h_s(n, 0.0);
    CHECK(cudaMemcpy(h_s.data(), S, sizeof(double) * n, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) {
      if (h_s[i] < -1e-12) {
        cleanup();
        PRINT_INPUT_ERROR("get_cholesky: matrix is not positive semidefinite.");
      }
    }

    std::vector<double> h_a(n * n, 0.0), h_l(n * n, 0.0);
    CHECK(cudaMemcpy(h_a.data(), A, sizeof(double) * n * n, cudaMemcpyDefault));
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j <= i; ++j) {
        double sum = h_a[i + j * n];
        for (int k = 0; k < j; ++k) {
          sum -= h_l[i + k * n] * h_l[j + k * n];
        }
        if (i == j) {
          if (sum <= 1e-14) {
            cleanup();
            PRINT_INPUT_ERROR("get_cholesky: matrix is not symmetric positive definite.");
          }
          h_l[i + j * n] = sqrt(sum);
        } else {
          h_l[i + j * n] = sum / h_l[j + j * n];
        }
      }
    }
    CHECK(cudaMemcpy(L, h_l.data(), sizeof(double) * n * n, cudaMemcpyDefault));
    GPU_CHECK_KERNEL;

    cleanup();
  }

  __global__ void gpu_sum(double* a, const int size, double* result)
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
    gpu_sum<<<1, 1024>>>(a.data(), a.size(), result.data());
    result.copy_to_host(&ret);
    return ret;
  }

  double sum(double* a, int size)
  {
    double ret;
    GPU_Vector<double> result(1);
    gpu_sum<<<1, 1024>>>(a, size, result.data());
    result.copy_to_host(&ret);
    return ret;
  }


  void sum2d(GPU_Vector<double>& a, double* result, int len, int nla=0)
  {
    int nl = (nla==0) ? a.size() / len : nla;
    GPU_Vector<double> temp(len * nla);
    GPU_Vector<double> d_result(len);
    temp.copy_from_device(a.data());
    for (int i=0;i<len;i++){
      gpu_sum<<<1, 1024>>>(&temp[i * nl], nl, &d_result[i]);
    }
    d_result.copy_to_host(result);
  }

  double __attribute__((unused)) dot(GPU_Vector<double>& a, GPU_Vector<double>& b)
  {
    GPU_Vector<double> temp(a.size());
    pairwise_product(a, b, temp);
    return sum(temp);
  }

  void scalar_multiply(GPU_Vector<double>& c, const double& a, GPU_Vector<double>& b)
  {
    int size = b.size();
    gpu_multiply<<<(size - 1) / 128 + 1, 128>>>(c.data(), a, b.data(), size);
  }

  __global__ void gpu_sum_square_axis1(double* dst, double* a, const int nl, const int ncol)
  {
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    double sum = 0;
    if (n < nl){
      for (int i = 0; i < ncol; i++){
        sum += a[n + i * nl] * a[n + i * nl];
      }
      dst[n] = sum;
    }
  }


  GPU_Vector<double> __attribute__((unused)) sum_square_axis1(
    GPU_Vector<double>& a, const int ncol)
  {
    int nl = a.size()/ncol;
    GPU_Vector<double> temp(nl);
    gpu_sum_square_axis1<<<(nl - 1) / 128 + 1, 128>>>(temp.data(), a.data(), nl, ncol);
    return temp;
  }

  double __attribute__((unused)) max_abs(int size, double* vec)
  {
    int index;
    double result;
    cublasIdamax(handle, size, vec, 1, &index);
    printf("max index: %d, ", index);
    cudaMemcpy(&result, vec + index - 1, sizeof(double), cudaMemcpyDeviceToHost);
    return abs(result);
  }

  double max_abs(int size, double* vec, int nsingle, bool printflag=false)
  {
    int index;
    double result;
    cublasIdamax(handle, size, vec, 1, &index);
    if (printflag){
      printf("i_fmax: %d:%d", index/nsingle, (index%nsingle)%int(nsingle/3));
      if ((index+9) % nsingle < 9) {printf("(D), ");} else {printf("(R), ");}
    }
    cudaMemcpy(&result, vec + index - 1, sizeof(double), cudaMemcpyDeviceToHost);
    return abs(result);
  }

  bool in_list(list<int>& mylist, int i){
    list<int>::iterator it = std::find(mylist.begin(), mylist.end(), i);
    if (it != mylist.end()) return true;
    else return false;
  }

  void print_setting(const char* name, int value){
    printf("%-20s = %d\n", name, value);
  }
  void print_setting(const char* name, bool value){
    printf("%-20s = %s\n", name, value?"true":"false");
  }
  void print_setting(const char* name, double value){
    printf("%-20s = %g\n", name, value);
  }
  void __attribute__((unused)) print_setting(const char* name, const char* value){
    printf("%-20s = %s\n", name, value);
  }
  void print_setting(const char* name, string value){
    printf("%-20s = %s\n", name, value.data());
  }

  struct is_greater_equal
  {
    int n_;
    is_greater_equal(int n){n_=n;}

    __host__ __device__
    bool operator()(int x)const {
      return x>=n_;
    }
  };

  // struct is_great
  // {
  //   __host__ __device__
  //   bool operator()(int x)const {
  //     return x>4;
  //   }
  // };
} // namespace


Spring::Spring(double k0, double de0, GPU_Vector<double> t0):k(k0),de(de0),t(t0)
  {
    cublasDnrm2(handle, t.size(), t.data(), 1, &nt);
  };

GPU_Vector<double> NormalTangentMethod::compute_tangent(Spring& spring1, Spring& spring2)
{
  GPU_Vector<double>& t1 = spring1.t;
  GPU_Vector<double>& t2 = spring2.t;
  int size = spring1.t.size();
  GPU_Vector<double> tangent(size);
  double nt;
  vector_add(tangent, t1, t2);
  cublasDnrm2(handle, size, tangent.data(), 1, &nt);
  scalar_multiply(tangent, 1/(nt+1e-10), tangent);
  return tangent;
}

void NormalTangentMethod::add_image_force(
  int size,
  double& tangential_force,
  double* tangent,
  Spring& spring1,
  Spring& spring2,
  double* imgforce)
{
  double scalar = -tangential_force + (spring2.nt*spring2.k - spring1.nt*spring1.k);
  cublasDaxpy_v2(handle, size, &scalar, tangent, 1, imgforce, 1);
}

GPU_Vector<double> ImprovedTangentMethod::compute_tangent(Spring& spring1, Spring& spring2)
{
  GPU_Vector<double>& t1 = spring1.t;
  GPU_Vector<double>& t2 = spring2.t;
  double de1 = spring1.de;
  double de2 = spring2.de;
  int size = spring1.t.size();
  GPU_Vector<double> tangent(size);
  double nt;
  // printf("de1=%f, de2=%f\n", de1, de2);
  // print_gpu(t1, "t1");
  // print_gpu(t2, "t2");
  // printf("nt1= %f, nt2= %f\n", nt1, nt2);
  if (de1 > 0 && de2 > 0) tangent.copy_from_device(t2.data());
  else if (de1 < 0 && de2 < 0) tangent.copy_from_device(t1.data());
  else{
    double scale1, scale2;
    double de_max = max(abs(de1), abs(de2));
    double de_min = min(abs(de1), abs(de2));
    tangent.fill(0.0);
    if (de2 + de1 > 0){
      scale1 = de_min / spring1.nt;
      scale2 = de_max / spring2.nt;
    }
    else{
      scale1 = de_max / spring1.nt;
      scale2 = de_min / spring2.nt;
    }
    // if (de2 + de1 > 0){
    //   scale1 = de_min;
    //   scale2 = de_max;
    // }
    // else{
    //   scale1 = de_max ;
    //   scale2 = de_min;
    // }
    cublasDaxpy(handle, size, &scale1, t1.data(), 1, tangent.data(), 1);
    cublasDaxpy(handle, size, &scale2, t2.data(), 1, tangent.data(), 1);
  }
  cublasDnrm2(handle, size, tangent.data(), 1, &nt);
  // printf("nt= %f\n", nt);
  scalar_multiply(tangent, 1/(nt+1e-10), tangent);
  // print_gpu(tangent, "tangent");
  return tangent;
}

void ImprovedTangentMethod::add_image_force(
  int size,
  double& tangential_force,
  double* tangent,
  Spring& spring1,
  Spring& spring2,
  double* imgforce)
{
  double scalar = -tangential_force + (spring2.nt*spring2.k - spring1.nt*spring1.k);
  cublasDaxpy_v2(handle, size, &scalar, tangent, 1, imgforce, 1);
}

void ModifiedImprovedTangentMethod::ensure_workspace(int size) {
  if (workspace_size == size) return;

  perp_force.resize(size);
  unit_perp_force.resize(size);
  ori_spring_force.resize(size);
  par_spring_force.resize(size);
  perp_spring_force.resize(size);
  dneb_force.resize(size);

  workspace_size = size;
}

void ModifiedImprovedTangentMethod::add_image_force(
  int size,
  double& tangential_force,
  double* tangent,
  Spring& spring1,
  Spring& spring2,
  double* imgforce)
{
  ensure_workspace(size);
  
  GPU_Vector<double>& t1 = spring1.t;
  GPU_Vector<double>& t2 = spring2.t;

  // ori_spring_force = k2 * t2 - k1 * t1
  double minus_k1 = -spring1.k;
  ori_spring_force.fill(0.0);
  cublasDaxpy(handle, size, &minus_k1, t1.data(), 1, ori_spring_force.data(), 1);
  cublasDaxpy(handle, size, &spring2.k, t2.data(), 1, ori_spring_force.data(), 1);

  // perp_force = imgforce - tangential_force * tangent
  perp_force.copy_from_device(imgforce);
  double minus_tf = -tangential_force;
  cublasDaxpy(handle, size, &minus_tf, tangent, 1, perp_force.data(), 1);

  // unit_perp_force = F_perp / |F_perp|
  double norm_pf;
  cublasDnrm2(handle, size, perp_force.data(), 1, &norm_pf);
  double inverse_norm_pf = 1.0 / (norm_pf + 1e-10);
  unit_perp_force.fill(0.0);
  cublasDaxpy(handle, size, &inverse_norm_pf, perp_force.data(), 1, unit_perp_force.data(), 1);

  // perp_spring_force = ori_spring_force - (ori_spring_force . tangent) * tangent
  perp_spring_force.copy_from_device(ori_spring_force.data());
  double dot_ot;
  cublasDdot(handle, size, ori_spring_force.data(), 1, tangent, 1, &dot_ot);
  par_spring_force.fill(0.0);
  cublasDaxpy(handle, size, &dot_ot, tangent, 1, par_spring_force.data(), 1);
  double minus_one = -1;
  cublasDaxpy(handle, size, &minus_one, par_spring_force.data(), 1, perp_spring_force.data(), 1);

  // F_dneb = perp_spring_force - (perp_spring_force . unit_perp_force) * unit_perp_force
  double dot_pu;
  cublasDdot(handle, size, perp_spring_force.data(), 1, unit_perp_force.data(), 1, &dot_pu);
  dneb_force.copy_from_device(perp_spring_force.data());
  double minus_dot_pu = -dot_pu;
  cublasDaxpy(handle, size, &minus_dot_pu, unit_perp_force.data(), 1, dneb_force.data(), 1);

  // F_swdneb = 2/pi * atan(|F_perp|^2 / |F_perp_spring|^2) * F_dneb
  double norm_psf;
  cublasDnrm2(handle, size, perp_spring_force.data(), 1, &norm_psf);
  double w = (2.0 / M_PI) * atan((norm_pf * norm_pf) / (norm_psf * norm_psf + 1e-20));
  cublasDaxpy(handle, size, &w, dneb_force.data(), 1, imgforce, 1);  //add F_swdneb

  double scalar = -tangential_force;
  cublasDaxpy(handle, size, &scalar, tangent, 1, imgforce, 1); //remove tangential force
  double one = 1;
  cublasDaxpy(handle, size, &one, par_spring_force.data(), 1, imgforce, 1); //add parallel spring force
}

NEB::NEB(){
  cublasCreate(&handle);
  cusolverDnCreate(&cusolverH);
}

NEB::~NEB() {
    cublasDestroy(handle);
    cusolverDnDestroy(cusolverH);
}

void NEB::parse_options(const char** param, int num_param, int& n)
{
  // Input Files: structure file names, initial path construction, and endpoint relaxation.
  if (strcmp(param[n], "is_name") == 0){
    istate_name.assign(param[n+1]);
    n++;
  } else if (strcmp(param[n], "fs_name") == 0){
    fstate_name.assign(param[n+1]);
    n++;
  } else if (strcmp(param[n], "has_mid") == 0){
    has_mid = true;
  } else if (strcmp(param[n], "mid_name") == 0){
    mid_name.assign(param[n+1]);
    n++;
  } else if (strcmp(param[n], "mid_name_list") == 0){
    for (int i=n+1; i<num_param; i++){
      mid_name_list.push_back(string(param[i]));
      n++;
      if (strcmp(param[n], "mid_name_list_end") == 0) break;
    }
    n++;
  } else if (strcmp(param[n], "suffix") == 0){
    string suffix(param[n+1]);
    istate_name.assign("is_"+suffix+".xyz");
    fstate_name.assign("fs_"+suffix+".xyz");
    mid_name.assign("mid_"+suffix+".xyz");
    n++;
  } else if (strcmp(param[n], "traj_name") == 0){
    traj_name.assign(param[n+1]);
    n++;
  } else if (strcmp(param[n], "interpolate") == 0){
    if (!is_valid_int(param[n+1], &n_interpolate)) {
      PRINT_INPUT_ERROR("interpolate should be an int.");
    }
    n++;
  } else if (strcmp(param[n], "need_relax") == 0){
    need_relax = true;

  // System Settings: cell degrees of freedom, pressure, rigid-body cleanup, and image spacing.
  } else if (strcmp(param[n], "no_vc") == 0){
    variable_cell = false;
  } else if (strcmp(param[n], "p") == 0){
    double press_scalar;
    if (!is_valid_real(param[n+1], &press_scalar)) {
      PRINT_INPUT_ERROR("p should be a real.");
    }
    pressure = {press_scalar};
    n++;
  } else if (strcmp(param[n], "p3") == 0){
    pressure.resize(3);
    for (int i=0; i<3; i++){
      if (!is_valid_real(param[n+1+i], &pressure[i])) {
        PRINT_INPUT_ERROR("p3 should be 3 reals.");
      }
    }
    n += 3;
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
    n += 6;
  } else if (strcmp(param[n], "remove_translation") == 0){
    remove_translation = true;
  } else if (strcmp(param[n], "remove_rotation") == 0){
    remove_rotation = true;
  } else if (strcmp(param[n], "dist_range") == 0){
    if (!is_valid_real(param[n+1], &min_dist) ||
        !is_valid_real(param[n+2], &max_dist)) {
      PRINT_INPUT_ERROR("dist_range should be two reals.");
    }
    n+=2;
  } else if (strcmp(param[n], "dist_ncount") == 0){
    if (!is_valid_int(param[n+1], &dist_ncount)) {
      PRINT_INPUT_ERROR("dist_ncount should be an int.");
    }
    n++;

  // NEB Method Settings: spring model, tangent choice, and special image treatment.
  } else if (strcmp(param[n], "k") == 0){
    if (!is_valid_real(param[n+1], &k)) {
      PRINT_INPUT_ERROR("k should be a real.");
    }
    n++;
  } else if (strcmp(param[n], "energy_based_k") == 0){
    energy_based_k = true;
  } else if (strcmp(param[n], "tangent") == 0){
    tangent_method_name = string(param[n+1]);
    n++;
  } else if (strcmp(param[n], "climb") == 0){
    climb = true;
  } else if (strcmp(param[n], "find_min") == 0){
    find_min = true;
  } else if (strcmp(param[n], "etol") == 0){
    if (!is_valid_real(param[n+1], &etol)) {
      PRINT_INPUT_ERROR("etol should be a real.");
    }
    n++;

  // Image Number Adjustment: controls for inserting/removing images along the path.
  } else if (strcmp(param[n], "no_ina") == 0){
    image_number_adjustment = false;
  } else if (strcmp(param[n], "ina_interval") == 0){
    if (!is_valid_int(param[n+1], &ina_interval)) {
      PRINT_INPUT_ERROR("ina_interval should be an int.");
    }
    n++;
  } else if (strcmp(param[n], "ina_k") == 0){
    ina_k = true;
  } else if (strcmp(param[n], "ina_cell_factor") == 0){
    if (!is_valid_real(param[n+1], &ina_cell_factor)) {
      PRINT_INPUT_ERROR("ina_cell_factor should be a real.");
    }
    n++;
  } else if (strcmp(param[n], "ina_force_tol") == 0){
    if (!is_valid_real(param[n+1], &ina_force_tol)) {
      PRINT_INPUT_ERROR("ina_force_tol should be a real.");
    }
    n++;
  } else if (strcmp(param[n], "ina_check_coord") == 0){
    if (!is_valid_int(param[n+1], &ina_check_coord)) {
      PRINT_INPUT_ERROR("ina_check_coord should be an int.");
    }
    n++;
  } else if (strcmp(param[n], "inacc_num") == 0){
    if (!is_valid_real(param[n+1], &inacc_num)) {
      PRINT_INPUT_ERROR("inacc_num should be a real.");
    }
    if (inacc_num < 0) PRINT_INPUT_ERROR("inacc_num should >= 0");
    n++;
  } else if (strcmp(param[n], "inacc_rc") == 0){
    if (!is_valid_real(param[n+1], &inacc_rc)) {
      PRINT_INPUT_ERROR("inacc_rc should be a real.");
    }
    if (inacc_rc <= 0) PRINT_INPUT_ERROR("inacc_rc should > 0");
    n++;

  // Output Settings: trajectory/energy snapshots and progress reporting.
  } else if (strcmp(param[n], "peek_interval") == 0){
    if (!is_valid_int(param[n+1], &peek_interval)) {
      PRINT_INPUT_ERROR("peek_interval should be an int.");
    }
    if (peek_interval <= 0) PRINT_INPUT_ERROR("peek_interval should > 0.");
    n++;
  } else if (strcmp(param[n], "dump_interval") == 0){
    if (!is_valid_int(param[n+1], &dump_interval)) {
      PRINT_INPUT_ERROR("dump_interval should be an int.");
    }
    if (dump_interval <= 0) PRINT_INPUT_ERROR("dump_interval should > 0.");
    n++;
  } else if (strcmp(param[n], "print_interval") == 0){
    if (!is_valid_int(param[n+1], &print_interval)) {
      PRINT_INPUT_ERROR("print_interval should be an int.");
    }
    if (print_interval <= 0) PRINT_INPUT_ERROR("print_interval should > 0.");
    n++;
  } else if (strcmp(param[n], "print_k") == 0){
    print_k = true;
  } else if (strcmp(param[n], "count_force_calc") == 0){
    count_force_calc = true;
  } else {
    string text="no keyword match with: ";
    text += param[n];
    PRINT_INPUT_ERROR(text.data());
  }
}

void NEB::parse_neb(const char** param, int num_param, Force& force)
{
  p_force = &force;

  if (strcmp(param[0], "neb_run") == 0) {
    if (strcmp(param[1], "fire") == 0) {
      minimizer_type = 1;
      if (num_param < 4) {
        PRINT_INPUT_ERROR("minimize fire should have at least 2 parameters.");
      }
      if (!is_valid_real(param[2], &force_tolerance)) {
        PRINT_INPUT_ERROR("Force tolerance should be a number.");
      }
      if (!is_valid_int(param[3], &max_steps)) {
        PRINT_INPUT_ERROR("Number of steps should be an integer.");
      }
      for (int n=4; n<num_param; n++){
        optimizer_opt.push_back(param[n]);
      }
      if (max_steps <= 0) {
        PRINT_INPUT_ERROR("Number of steps should > 0.");
      }
      printf("\nStart to do neb calculation.\n");
      printf("    using the fast inertial relaxation engine (FIRE) method.\n");
      printf("    with a force tolerance of %g eV/A.\n", force_tolerance);

      run_neb();
    }
  } else if (strcmp(param[0], "neb_set") == 0){
    for (int n=1; n<num_param; n++){
      parse_options(param, num_param, n);
    }
  }
  
}

void NEB::reset_minimizer(int number_of_atoms, int max_steps, double force_tolerance) {
  switch (minimizer_type) {
  case 1:
    printf("----------------------------------------\n");
    printf("New minimization, maximally %d steps.\n", max_steps);

    minimizer.reset(new Minimizer_FIRE_JQH(number_of_atoms, max_steps, force_tolerance));
    dynamic_cast<Minimizer_FIRE_JQH&>(*minimizer).parse_FIRE(
      optimizer_opt.data(), optimizer_opt.size(), 0, true?step==0:false);
    break;
  default:
    PRINT_INPUT_ERROR("Invalid minimizer.");
    break;
  }
}

std::unique_ptr<BaseTangentMethod> get_tangent_method(string tangent_method_name, double k){
  if (tangent_method_name == string("improved")){
    return make_unique<ImprovedTangentMethod>(k);
  } else if (tangent_method_name == string("modified")){
    return make_unique<ModifiedImprovedTangentMethod>(k);
  } else if (tangent_method_name == string("normal")){
    return make_unique<NormalTangentMethod>(k);
  } else {
     printf("No tangent method match with: %s\n", tangent_method_name.data());
     printf("Valid Options: improved, normal\n");
     exit(-1);
  }
}

void cell_best_match(double* cell_ref, double* cell, double* new_cell){
  double *H, *rot;
  cudaMalloc(&H, 9*sizeof(double));
  cudaMalloc(&rot, 9*sizeof(double));
  // gpu_matmul(cell_ref, cell, H, 3, 3, 3, 1, 0);
  // gpu_matmul(rot, cell, new_cell, 3, 3, 3);


}


void NEB::initialize_images() {
  if (traj_name.size() != 0){
    printf("--------------file %s to traj-------------------\n", traj_name.data());
    ifstream input(traj_name);
    bool read_success;
    if (!variable_cell){
      while (true){
        Atoms* p_tmp = new Atoms(input, read_success);
        if (read_success) {
          // mid_list.push_back(make_pair(-1, p_tmp));
          images.emplace_back(unique_ptr<Atoms>(p_tmp));
        } else break;
      }
    }
    else {
      Atoms* p_is = new Atoms(input, read_success);
      if (read_success) {
        images.push_back(make_unique<VCWrapper>(*p_is, pressure));
      } else {
        printf("read traj failed\n");
        exit(-1);
      }
      h_ref.assign(p_is->box.cpu_h, p_is->box.cpu_h+9);
      while (true){
        VCWrapper* p_tmp = new VCWrapper(input, read_success, pressure, h_ref.data());
        if (read_success) {
          // mid_list.push_back(make_pair(-1, p_tmp->get_p_atoms()));
          images.push_back(unique_ptr<VCWrapper>(p_tmp));
        } else break;
      }
    }
    imid_list.push_back(-1);
    printf("traj nimages: %d\n", int(images.size()));
    input.close();
  }
  else {
    printf("midname: %s\n", mid_name.data());
    Atoms *p_is = new Atoms(istate_name.data());
    Atoms *p_fs = new Atoms(fstate_name.data());
    h_ref.assign(p_is->box.cpu_h, p_is->box.cpu_h+9);
    // GPU_Vector<double> tmp_h = 9, tmp_h2(9);
    // tmp_h.copy_from_host(h_ref.data());
    // tmp_h2.copy_from_host(p_fs->box.cpu_h);
    // print_gpu(tmp_h, "tmp_h");
    // print_gpu(tmp_h2, "tmp_h2");
    // cell_best_match(tmp_h.data(), tmp_h2.data(), tmp_h2.data());
    // print_gpu(tmp_h2, "tmp_h2");
    if (mid_name_list.size() == 0) mid_name_list.push_back(mid_name);
    // print_arr(h_ref.data(), 9, "vector h_ref");
    if (!variable_cell){
      images.push_back(unique_ptr<Atoms>(p_is));
      if (has_mid){
        for (int i=0; i<mid_name_list.size(); i++){
          Atoms *p_tmp = new Atoms((mid_name_list[i]).data());
          images.push_back(unique_ptr<Atoms>(p_tmp));
          imid_list.push_back((i+1)*n_interpolate/(mid_name_list.size()+1));
          // mid_list.push_back(make_pair((i+1)*n_interpolate/(mid_name_list.size()+1) + 1, p_tmp));
        }
      }
      images.push_back(unique_ptr<Atoms>(p_fs));
    } else{
      images.push_back(make_unique<VCWrapper>(*p_is, pressure, h_ref.data()));
      if (has_mid){
        for (int i=0; i<mid_name_list.size(); i++){
          Atoms *p_tmp = new VCWrapper((mid_name_list[i]).data(), pressure, h_ref.data());
          images.push_back(unique_ptr<Atoms>(p_tmp));
          imid_list.push_back((i+1)*n_interpolate/(mid_name_list.size()+1));
          // mid_list.push_back(make_pair((i+1)*n_interpolate/(mid_name_list.size()+1) + 1, p_tmp));
        }
      }
      images.push_back(make_unique<VCWrapper>(*p_fs, pressure, h_ref.data()));
    }
  }
  natoms_per_image = images[0]->get_natoms();
  n_realatoms = images[0]->get_p_atoms()->get_natoms();
  optimize_factor = pow(n_realatoms, 1.0/4);
  printf("optimize_factor=%f\n", optimize_factor);
}

void NEB::run_neb() {
  initialize_images();
  tangentmethod = get_tangent_method(tangent_method_name, k);
  dist_ncount = (dist_ncount < n_realatoms) ? dist_ncount : n_realatoms;
  if (dump_interval == -1) dump_interval = (max_steps - 1) / 10 + 1;
  if (peek_interval == -1) peek_interval = (max_steps - 1) / 50 + 1;
  if (ina_cell_factor == -1.0) ina_cell_factor = pow(n_realatoms, 1.0/6);

  printf("-----------------neb settings-----------------\n");
  print_setting("k", k);
  print_setting("energy_based_k", energy_based_k);
  print_setting("variable_cell", variable_cell);
  if (variable_cell){
    printf("%-20s =", "pressure");
    for (auto x:pressure) printf(" %.4f", x);
    printf("\n");
  }
  print_setting("climb", climb);
  print_setting("find_min", find_min);
  if (climb) print_setting("etol", etol);
  print_setting("image_number_adjustment", image_number_adjustment);
  if (image_number_adjustment) {
    print_setting("ina_interval", ina_interval);
    print_setting("min_dist", min_dist);
    print_setting("max_dist", max_dist);
    print_setting("ina_cell_factor", ina_cell_factor);
    print_setting("dist_ncount", dist_ncount);
    print_setting("ina_check_coord", ina_check_coord);
    if (ina_check_coord) {
      print_setting("inacc_num", inacc_num);
      print_setting("inacc_rc", inacc_rc);
    }
  }
  print_setting("has_mid", has_mid);
  if (has_mid) print_setting("n_interpolate", n_interpolate);
  print_setting("need_relax", need_relax);
  print_setting("remove_translation", remove_translation);
  print_setting("remove_rotation", remove_rotation);
  print_setting("tangent_method", tangent_method_name);
  print_setting("max_steps", max_steps);
  print_setting("dump_interval", dump_interval);
  print_setting("peek_interval", peek_interval);
  print_setting("print_interval", print_interval);
  printf("----------------------------------------------\n");

  if (inacc_num < 1) inacc_num *= n_realatoms;
  if (etol < 0) etol *= -n_realatoms;
  // printf("force id: %s, nep id: %s\n",typeid(*p_force->potentials[0]).name(), typeid(NEP3).name());
  // -----reinitialize nep to make sure that natom in it is right------
  if (typeid(*(p_force->potentials[0]))==typeid(NEP)){
    printf("nep forces\n");
    dynamic_cast<NEP&>(*p_force->potentials[0]).resize(n_realatoms);
  }
  for (int i=0; i < images.size(); i++) images[i]->set_calc(*p_force);
  if (need_relax){
    printf("--------------relax-------------\n");
    double relax_tol=min(0.001, force_tolerance);
    reset_minimizer(natoms_per_image, 10000, relax_tol);
    minimizer->compute(*images.front());
    reset_minimizer(natoms_per_image, 10000, relax_tol);
    minimizer->compute(*images.back());
    printf("-----------relax finish---------\n");
    FILE* fid=fopen("relaxed_is_fs.xyz", "w");
    Atoms& atoms_is = *images.front()->get_p_atoms();
    save_one_frame(fid, atoms_is.box, atoms_is.get_energy(), images.front()->get_energy(), atoms_is.cpu_atom_symbol,
       atoms_is.get_positions());
    Atoms& atoms_fs = *images.back()->get_p_atoms();
    save_one_frame(fid, atoms_fs.box, atoms_fs.get_energy(), images.back()->get_energy(), atoms_fs.cpu_atom_symbol,
       atoms_fs.get_positions());
    fclose(fid);
  }
  if (remove_translation){
    double center[3], ref_center[3];
    ref_center[0] = (h_ref[0] + h_ref[1] + h_ref[2])/2;
    ref_center[1] = (h_ref[3] + h_ref[4] + h_ref[5])/2;
    ref_center[2] = (h_ref[6] + h_ref[7] + h_ref[8])/2;
    for (auto it=images.begin();it!=images.end();it++){
      GPU_Vector<double>& pos = (*it)->get_positions();
      sum2d(pos, center, 3, n_realatoms);
      // print_arr(ref_center, 3, "ref_center");
      // print_arr(center, 3, "center");
        // print_gpu(pos, "pos_0");
      for (int i=0;i<3;i++){
        center[i] /= n_realatoms;
        // printf("center %d: %f, ref_center[i]-center[i]:%f\n", i, center[i], ref_center[i]-center[i]);
        gpu_vector_add_scalar<<<(n_realatoms-1)/128+1,128>>>
            (pos.data() + i*n_realatoms, pos.data() + i*n_realatoms, ref_center[i]-center[i], n_realatoms);
        // printf("meanpos: %f\n", sum(pos.data() + i*n_realatoms, n_realatoms)/n_realatoms);
      }
      (*it)->set_positions();
        // print_gpu(pos, "pos_1");
    }
  }
  if (n_interpolate > 0){
    interpolate();
  }
  if (images.size() <= 2){
    printf("There should be at least one intermediate image.\n");
    exit(1);
  }
  for (int i=0; i < images.size(); i++) images[i]->set_calc(*p_force);
  #ifdef DEBUG
  printf("run_neb() images[0] natoms %d\n", images[0]->get_natoms());
  #endif

  images.front()->compute();
  images.back()->compute();
  first_energy = images.front()->get_energy();
  last_energy = images.back()->get_energy();
  
  klist.resize(images.size() - 1, k);

  double fnrm2; // used to check if minimization is finished or nimages changes
  // -------------------------main loop------------------------------
  while (true){
    initialize_compute();
    reset_minimizer(natoms, max_steps - step, force_tolerance);
    minimizer->compute(*this);
    // printf("neb total steps: %d\n", step);
    if (ina_count != 0) write_energies();
    cublasDnrm2(handle, natoms_per_image*3, forces.data(), 1, &fnrm2);
    if (fnrm2 != 0.0) {
      // minimizer->reset_number_of_atoms((images.size()-2) * natoms_per_image);
      break;
    }
  }
  write_neb_traj("final_traj.xyz", "w");
}


void NEB::compute()
{
  // printf("neb compute\n");
  // compute original forces
  if (variable_cell){
    for (int i=1; i < nimages - 1; i++){
      // &forces[(i-1) * natoms_per_image*3]
      gpu_multiply<<<1, 9>>>(positions.data() + i*natoms_per_image*3 - 9,
            optimize_factor, positions.data() + i*natoms_per_image*3 - 9, 9);
    }
  }
  set_positions();
  for (int i=1; i < nimages - 1; i++){
    // printf("image %d\n", i);
    images[i]->compute();
    images[i]->get_forces().copy_to_device(
      &forces[(i-1) * natoms_per_image*3],
      natoms_per_image*3);
    gpu_multiply<<<1, 9>>>(forces.data() + i*natoms_per_image*3 - 9,
          optimize_factor, forces.data() + i*natoms_per_image*3 - 9, 9);
    // image_energies[i] = sum(images[i]->get_potential_per_atom());
    image_energies[i] = images[i]->get_energy();
  }
  n_force_calc += nimages - 2;

  if (step % dump_interval == 0 && step != 0) write_neb_traj("dump_traj.xyz", "a");
  
  if (step % peek_interval == 0 && step != 0){
    write_neb_traj("peek_traj.xyz", "w");
    write_energies();
  }

  find_min_max(etol);
  vector<double> k_effective_list(klist);
  // printf("klist: ");
  if (energy_based_k) {
    for (int i=0; i<nimages-1;i++){
      int dist2imaxes=nimages;
      for (auto x:imaxes) {
        if (abs(i-x) < dist2imaxes) dist2imaxes = abs(i-x);
        if (abs(i+1-x) < dist2imaxes) dist2imaxes = abs(i+1-x);
      }
      double energy_based_k_factor = 1.0 / (1.0 - 0.8*pow(0.9, pow(dist2imaxes,2)));
      k_effective_list[i] *= energy_based_k_factor;
      // printf("%.3f ", k_effective_list[i]);
    }
  }
  // printf("\n"); 

  // -----------------start to compute spring force----------------------
  // GPU_Vector<double> tangent(natoms_per_image*3);
  GPU_Vector<double> t1(natoms_per_image*3);
  GPU_Vector<double> t2(natoms_per_image*3);
  GPU_Vector<double> spring_force(natoms_per_image*3);
  vector_substract(t1, images[1]->get_positions(), images[0]->get_positions());
  Spring spring1{k_effective_list[0], image_energies[1] - image_energies[0], t1};
  
  for (int i=1; i < nimages - 1; i++){
    vector_substract(t2, images[i+1]->get_positions(), images[i]->get_positions());
    Spring spring2{k_effective_list[i], image_energies[i+1] - image_energies[i], t2};
    // print_gpu(t1, "t1");
    GPU_Vector<double> tangent = tangentmethod->compute_tangent(spring1, spring2);
    // print_gpu(tangent, "t");
    double tangential_force;
    cublasDdot(handle, 3*natoms_per_image, &forces[(i-1)*natoms_per_image*3], 1,
     tangent.data(), 1, &tangential_force);
    // print_gpu(tangential_force, "tangential_force");
    if (climb && in_list(imaxes, i)){
      double tmp_num = -2.0 * tangential_force;
      cublasDaxpy(handle, natoms_per_image*3, &tmp_num,
        tangent.data(), 1, &forces[(i-1)*natoms_per_image*3], 1);
    } else if (find_min && in_list(imins, i)){
      ;
    }
    else{
      tangentmethod->add_image_force(natoms_per_image*3,
        tangential_force, tangent.data(), spring1, spring2,
        &forces[(i-1)*natoms_per_image*3]);
    }
      
    if (remove_translation){
      double mean_force;
      for (int j=0;j<3;j++){
        mean_force = sum(forces.data() + 3*(i-1)*natoms_per_image + j*n_realatoms, n_realatoms);
        mean_force /= n_realatoms;
        gpu_vector_add_scalar<<<(natoms-1)/128+1,128>>>(
          forces.data() + 3*(i-1)*natoms_per_image + j*n_realatoms,
          forces.data() + 3*(i-1)*natoms_per_image + j*n_realatoms,
          -mean_force, n_realatoms);
      }
    }
    if (remove_rotation){
      // printf("remove rot\n");
      GPU_Vector<double> virial_real(9, Memory_Type::managed), cur_deform(18, Memory_Type::managed);
      cur_deform.copy_from_device(positions.data() + 3*i*natoms_per_image-9, 9);
      get_3x3_inverse(cur_deform.data(), cur_deform.data() + 9);
      // print_gpu(cur_deform);
      gpu_matmul(
        positions.data() + 3*i*natoms_per_image-9,
        forces.data() + 3*i*natoms_per_image-9,
        virial_real.data(),
        3, 3, 3, 1, 0);
      GPU_CHECK_KERNEL;
      cudaDeviceSynchronize();
      // print_gpu(virial_real, "vr1");
      virial_real[1] = virial_real[3] = 0.5 * (virial_real[1] + virial_real[3]);
      virial_real[2] = virial_real[6] = 0.5 * (virial_real[2] + virial_real[6]);
      virial_real[5] = virial_real[7] = 0.5 * (virial_real[5] + virial_real[7]);
      // print_gpu(virial_real, "vr2");
      gpu_matmul(
        cur_deform.data() + 9,
        virial_real.data(),
        forces.data() + 3*i*natoms_per_image-9,
        3, 3, 3, 1, 0);
      GPU_CHECK_KERNEL;
    }
    spring1 = move(spring2);
  GPU_CHECK_KERNEL;
  }
  print_info();
  if (image_number_adjustment) adjust_image_number();
  if (variable_cell){
    for (int i=1; i < nimages - 1; i++){
      // &forces[(i-1) * natoms_per_image*3]
      gpu_multiply<<<1, 9>>>(forces.data() + i*natoms_per_image*3 - 9,
            1/optimize_factor, forces.data() + i*natoms_per_image*3 - 9, 9);
      gpu_multiply<<<1, 9>>>(positions.data() + i*natoms_per_image*3 - 9,
            1/optimize_factor, positions.data() + i*natoms_per_image*3 - 9, 9);
    }
  }
  if (ina_count==0){
    if (print_k) print_arr(k_effective_list.data(), k_effective_list.size(), "k_effective_list");
    forces.fill(0);
    printf("imaxes before change: ");
    for_each(imaxes.begin(), imaxes.end(), [](int a){printf("%d ", a);});
    printf("\n");
  }
  step++;
  // print_gpu(forces, "neb forces");
  // print_gpu(positions, "neb pos");
}

void NEB::print_info(){
  auto it_max_energy = max_element(image_energies.begin(), image_energies.end());
  cudaDeviceSynchronize();
  potential_per_atom[0] = *it_max_energy - first_energy;
  if (step % print_interval == 0){
    printf("step: %d, ", step);
    fmax = max_abs(natoms*3, forces.data(), natoms_per_image*3, true);
    printf("emax= %f(%d), ", *it_max_energy - first_energy, int(it_max_energy-image_energies.begin()));
    printf("fmax=%f\n",fmax);
    if (count_force_calc) printf("INA info: %d\t%d\t%d\t%f\n", step, nimages, n_force_calc, fmax);
  } else {
    fmax = max_abs(natoms*3, forces.data(), natoms_per_image*3, false);
  }
}

void NEB::adjust_image_number() {
  // printf("adjust_image_number, natoms: %d, forces.size: %d\n", natoms, forces.size());
  fflush(stdout);
  if (ina_count < ina_interval || (ina_count < ina_interval *2 && fmax > 2) ||
      (ina_count < ina_interval *5 && fmax > 3) || fmax > 5){
    ina_count++;
    return;
  }
  GPU_Vector<double> dpos(natoms_per_image*3), new_pos(natoms_per_image*3);
  double dist;
  int max_neighbor = 10, high_coord_atom_count;
  GPU_Vector<int> cell_count(n_realatoms), cell_count_sum(n_realatoms), cell_contents(n_realatoms);
  GPU_Vector<int> NN(n_realatoms), NL(n_realatoms * max_neighbor);
  // for (auto it = images.begin()+1; it != images.end()-1; it++)
  // printf("dist:");
  // printf("image_dist: ");
  int i_ori = 0;
  for (int i = 1; i < images.size(); i++)
  {
    double cur_min_dist(min_dist), cur_max_dist(max_dist);
    int cur_dist_ncount(dist_ncount);
    i_ori++;
    if (ina_check_coord != 0.0){
      bool small_box = false;
      if (small_box){ // TODO

      }
      else {
        find_neighbor(
          0, n_realatoms, inacc_rc,
          images[i]->get_p_atoms()->box,
          images[i]->get_p_atoms()->type,
          images[i]->get_p_atoms()->get_positions(),
          cell_count, cell_count_sum, cell_contents,
          NN, NL
        );
      }
      thrust::device_ptr<int> d_ptr = thrust::device_pointer_cast(NN.data());
      high_coord_atom_count =
        thrust::count_if(d_ptr, d_ptr + n_realatoms, is_greater_equal(ina_check_coord));
      if (high_coord_atom_count > inacc_num){
        // printf("high_coord_atom_count = %d\n", high_coord_atom_count);
        // cur_min_dist *= 3;
        // cur_max_dist *= 3;
        cur_dist_ncount *= 3;
        cur_dist_ncount = (cur_dist_ncount < n_realatoms) ? cur_dist_ncount : n_realatoms;
      }
    };

    GPU_Vector<double>& pos1 = images[i-1]->get_positions();
    GPU_Vector<double>& pos2 = images[i]->get_positions();
    vector_add(dpos, pos2, pos1, 1.0, -1.0);

    // only count the largest dist_ncount displacements
    // GPU_Vector<double> temp(nl);
    GPU_Vector<double> r2_arr(n_realatoms), h2_arr(3);
    gpu_sum_square_axis1<<<(n_realatoms - 1) / 128 + 1, 128>>>(r2_arr.data(), dpos.data(), n_realatoms, 3);
    gpu_sum_square_axis1<<<1, 3>>>(h2_arr.data(), dpos.data() + 3*n_realatoms, 3, 3);
    thrust::device_ptr<double> d_ptr = thrust::device_pointer_cast(r2_arr.data());
    thrust::sort(d_ptr, d_ptr + n_realatoms);
    double h_sum_square = (variable_cell) ? sum(h2_arr.data(), 3) : 0;
    double r_sum_square = sum(r2_arr.data()+n_realatoms - cur_dist_ncount, cur_dist_ncount);
    double r_dist = sqrt(r_sum_square/cur_dist_ncount);
    double h_dist = sqrt(h_sum_square/n_realatoms) * ina_cell_factor;
    dist = r_dist + h_dist;

    if (dist > cur_max_dist){
      vector_add(new_pos, pos1, pos2, 0.5, 0.5);
      double k_old = klist[i-1];
      if (variable_cell){
        images.insert(images.begin()+i, make_unique<VCWrapper>(images[0].get(), new_pos.data()));
      } else {
        images.insert(images.begin()+i, make_unique<Atoms>(images[0].get(), new_pos.data()));
      }
      klist.insert(klist.begin() + i, k_old);
      if (ina_k) {
        double k_new = k_old * ina_k_efficient;
        if (k_new > k * 10) k_new = k_old;
        klist[i-1] = k_new;
        klist[i] = k_new;
      }
      printf("add an image: %d, nimages: %d, dist: %.6f(r), %.6f(h)\n",
        i_ori, int(images.size()), r_dist, h_dist);
      i+=2; //skip 2 images
      i_ori++;
      ina_count = 0;
    }else if (dist < cur_min_dist && i != images.size()-1){
      double k_old = klist[i-1];
      images.erase(images.begin()+i);
      klist.erase(klist.begin()+i);
      if (ina_k) {
        double k_new = k_old / ina_k_efficient;
        if (k_new < k / 10) k_new = k_old;
        klist[i-1] = k_new;
      }
      printf("remove an image: %d , nimages: %d\n", i_ori, int(images.size()));
      // i doesn't change, skip 2 images
      i_ori++;
      ina_count = 0;
    }
    if (ina_k) { //renomalize klist
      double avg_k = accumulate(klist.begin(), klist.end(), 0.0) / klist.size();
      for (int j=0; j<klist.size(); j++){
        klist[j] = klist[j] / avg_k * k;
      }
    }
  }
}

void NEB::write_neb_traj(const char* filename, const char* mode){
  printf("==================write %s==================\n", filename);
  FILE* fid=fopen(filename, mode);
  vector<double> cpu_positions(natoms_per_image*3);
  // vector<int> cpu_type((*images[0]->get_p_atoms()).type.size());
  // (*images[0]->get_p_atoms()).type.copy_to_host(cpu_type.data());
  for (int i=0;i<images.size();i++){
    Atoms& atoms = *images[i]->get_p_atoms();
    save_one_frame(fid, atoms.box, atoms.get_energy(), images[i]->get_energy(), atoms.cpu_atom_symbol,
       atoms.get_positions(), cpu_positions);
    // print_gpu(atoms.get_positions());
  }
  fclose(fid);
}

void NEB::interpolate() {
  // printf("neb interpolate, size of images[0]->get_positions().size()=%d\n", images[0]->get_positions().size());
  GPU_Vector<double> dpos(images[0]->get_positions().size()), cur_pos(images[0]->get_positions().size());
  vector<int> i_keyframe={0};
  vector<Atoms*> keyframe={images.front().get()};
  int n_key=1;
  bool equal_spacing = (imid_list.size() == 0 or imid_list.front() == -1);
  int n_mid = images.size() - 2;
  printf("nmid %d\n", n_mid);
  if (equal_spacing){
    for (int i=0; i< n_mid; i++){
      i_keyframe.push_back((i+1)*n_interpolate/(n_mid+1) + n_key);
      keyframe.push_back(images[n_key].get());
      n_key++;
    }
  }
  else {
    for (auto& imid : imid_list){
      i_keyframe.push_back(imid+n_key);
      keyframe.push_back(images[n_key].get());
      n_key++;
      // images.insert(images.begin()+n_key, it->second);
    }
  }
  i_keyframe.push_back(n_interpolate + n_key++);
  printf("nkey = %d\n", n_key);
  keyframe.push_back(images.back().get());
  print_arr(i_keyframe.data(), i_keyframe.size(), "i_k");  
  for (int k=0; k<(n_key-1);k++){
    GPU_Vector<double>& ipos = keyframe[k]->get_positions();
    GPU_Vector<double>& fpos = keyframe[k+1]->get_positions();
    vector_add(dpos, fpos, ipos, 1.0, -1.0);
    int n_cur = i_keyframe[k+1] - i_keyframe[k];
    for (int i_cur=1;i_cur<n_cur;i_cur++){
      printf("k=%d, i_cur=%d, nimages=(%d)%d\n",
        k, i_cur, i_keyframe[k]+i_cur, int(images.size()));
      vector_add(cur_pos, ipos, dpos, 1, double(i_cur)/(n_cur));
      if (variable_cell){
        // VCWrapper* new_vcatoms = new VCWrapper(images[0].get(), cur_pos.data());
        images.insert(images.begin()+i_keyframe[k]+i_cur,
          make_unique<VCWrapper>(images[0].get(), cur_pos.data()));
      } else {
        // Atoms* new_atoms = new Atoms(*images[0].get(), cur_pos.data());
        images.insert(images.begin()+i_keyframe[k]+i_cur,
          make_unique<Atoms>(images[0].get(), cur_pos.data()));
      }
    }
  }
  write_neb_traj("interpolate.xyz", "w");
  // printf("neb interpolate finish\n");
}

void NEB::initialize_compute() {
  #ifdef DEBUG
  printf("neb initialize\n");
  #endif
  nimages = images.size();
  printf("nimages: %d, natoms_per_image: %d\n", nimages, natoms_per_image);
  natoms = (nimages - 2) * natoms_per_image; // remove first and last images

  potential_per_atom.resize(1, Memory_Type::managed);
  image_energies.resize(nimages);
  positions.resize(natoms * 3);
  forces.resize(natoms * 3, 0);

  build_positions();
  image_energies.front() = first_energy;
  image_energies.back() = last_energy;

}


void NEB::find_min_max(double etol)
{
  vector<int> extrema;
  extrema.reserve(nimages);
  for (int i = 1; i < nimages - 1; i++) {
    if (image_energies[i] > image_energies[i - 1] &&
        image_energies[i] > image_energies[i + 1]) {
      extrema.push_back(i);
    } else if (image_energies[i] < image_energies[i - 1] &&
               image_energies[i] < image_energies[i + 1]) {
      extrema.push_back(i);
    }
  }

  // Iteratively remove the closest adjacent extrema pair if their energy gap < etol.
  // Example: extrema energies [100, 96, 97, 92], etol=5 -> remove [96, 97], keep 100.
  if (etol > 0.0) {
    while (extrema.size() >= 2) {
      size_t best_pair = extrema.size();
      double best_diff = etol;
      for (size_t i = 0; i + 1 < extrema.size(); ++i) {
        double diff = abs(image_energies[extrema[i]] - image_energies[extrema[i + 1]]);
        if (diff < best_diff) {
          best_diff = diff;
          best_pair = i;
        }
      }
      if (best_pair == extrema.size()) {
        break;
      }
      extrema.erase(extrema.begin() + best_pair, extrema.begin() + best_pair + 2);
    }
  }

  imaxes.clear();
  imins.clear();
  for (const auto idx : extrema) {
    if (image_energies[idx] > image_energies[idx - 1] &&
        image_energies[idx] > image_energies[idx + 1]) {
      imaxes.push_back(idx);
    } else if (image_energies[idx] < image_energies[idx - 1] &&
               image_energies[idx] < image_energies[idx + 1]) {
      imins.push_back(idx);
    }
  }
  imax = max_element(image_energies.begin(), image_energies.end()) - image_energies.begin();
}

GPU_Vector<double>& NEB::build_positions()
{
  #ifdef DEBUG
  printf("neb build_position\n");
  #endif
  for (int i=1; i<nimages - 1; i++){
    images[i]->get_positions().copy_to_device(
      &positions[(i-1) * natoms_per_image*3],
      natoms_per_image*3);
  }
  for (int i=1; i < nimages - 1; i++){
    gpu_multiply<<<1, 9>>>(positions.data() + i*natoms_per_image*3 - 9,
          1/optimize_factor, positions.data() + i*natoms_per_image*3 - 9, 9);
  }
  return positions;
}

void NEB::set_positions()
{
  // printf("neb set_position\n");
  for (int i=1; i<nimages-1;i++){
    images[i]->get_positions().copy_from_device(
      &positions[(i-1) * natoms_per_image*3],
      natoms_per_image*3);
  }
}

void NEB::write_energies() {
  FILE* fid = fopen("neb_energies.out", "w");
  
  printf("        image_energies:");
  for (int i=0;i<image_energies.size();i++){
    if (i%10==0) printf("\n");
    double cur_energy = image_energies[i] - first_energy;
    
    if (in_list(imaxes, i)) printf("<%.3f>", cur_energy);
    else if (in_list(imins, i)) printf("(%.3f)", cur_energy);
    else printf(" %.3f ", cur_energy);
    fprintf(fid, "%.5f\n", image_energies[i] - first_energy);
  }
  double max_energy = *max_element(image_energies.begin(), image_energies.end());
  cudaDeviceSynchronize();
  potential_per_atom[0] = max_energy;
  printf("\n    Emax=%f, Ei=%f, Ef=%f\n", max_energy, max_energy-first_energy, max_energy-last_energy);

  fclose(fid);
}

double NEB::get_energy()
{ 
  return potential_per_atom[0];
}
