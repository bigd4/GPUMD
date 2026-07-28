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

  __global__ void gpu_image_force_max(
    const int atoms_per_image,
    const int real_atoms_per_image,
    const int number_of_images,
    const double cell_metric_scale,
    const double* force,
    double* force_max)
  {
    const int image = blockIdx.x;
    if (image >= number_of_images) return;

    const int image_size = atoms_per_image * 3;
    const int image_offset = image * image_size;
    double local_max = 0.0;
    for (int n = threadIdx.x; n < image_size; n += blockDim.x) {
      double value = abs(force[image_offset + n]);
      if (n >= real_atoms_per_image * 3) value /= cell_metric_scale;
      local_max = max(local_max, value);
    }

    __shared__ double block_max[256];
    block_max[threadIdx.x] = local_max;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
      if (threadIdx.x < offset) {
        block_max[threadIdx.x] = max(block_max[threadIdx.x], block_max[threadIdx.x + offset]);
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) force_max[image] = block_max[0];
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

  bool compare_image(Atoms& atoms1, Atoms& atoms2, int n_realatoms, bool variable_cell, double tolerance)
  {
    vector<double> pos1(atoms1.get_positions().size());
    vector<double> pos2(atoms2.get_positions().size());
    atoms1.get_positions().copy_to_host(pos1.data());
    atoms2.get_positions().copy_to_host(pos2.data());

    const int real_size = 3 * n_realatoms;
    for (int i = 0; i < real_size; ++i) {
      if (abs(pos1[i] - pos2[i]) > tolerance) return false;
    }

    if (variable_cell) {
      double cell_rssd2 = 0.0;
      for (int i = real_size; i < real_size + 9; ++i) {
        double diff = pos1[i] - pos2[i];
        cell_rssd2 += diff * diff;
      }
      if (sqrt(cell_rssd2) > 3.0 * tolerance) return false;
    }

    return true;
  }

  unique_ptr<Atoms> make_cell_filter(
    Atoms& atoms, const vector<double>& pressure, double* h_ref,
    bool rotation_free, double cell_factor)
  {
    if (rotation_free) {
      return make_unique<RotationFreeVCWrapper>(atoms, pressure, h_ref, cell_factor);
    }
    return make_unique<VCWrapper>(atoms, pressure, h_ref, cell_factor);
  }

  unique_ptr<Atoms> make_cell_filter(
    Atoms& atoms, const vector<double>& pressure, bool rotation_free, double cell_factor)
  {
    if (rotation_free) {
      return make_unique<RotationFreeVCWrapper>(atoms, pressure, cell_factor);
    }
    return make_unique<VCWrapper>(atoms, pressure, cell_factor);
  }

  unique_ptr<Atoms> make_cell_filter_from_position(
    Atoms* prototype, double* position, bool rotation_free)
  {
    if (rotation_free) {
      return make_unique<RotationFreeVCWrapper>(prototype, position);
    }
    return make_unique<VCWrapper>(prototype, position);
  }

  Atoms* new_cell_filter(
    const char* filename, const vector<double>& pressure, double* h_ref,
    bool rotation_free, double cell_factor)
  {
    if (rotation_free) {
      return new RotationFreeVCWrapper(filename, pressure, h_ref, cell_factor);
    }
    return new VCWrapper(filename, pressure, h_ref, cell_factor);
  }

  void align_image_by_mic(Atoms& ref_image, Atoms& image)
  {
    Atoms& ref_atoms = *ref_image.get_p_atoms();
    Atoms& atoms = *image.get_p_atoms();
    const int natoms = ref_atoms.get_natoms();
    if (atoms.get_natoms() != natoms) {
      PRINT_INPUT_ERROR("find_mic requires the same atom count in every NEB image.");
    }

    vector<double> ref_pos(ref_atoms.get_positions().size());
    vector<double> pos(atoms.get_positions().size());
    ref_atoms.get_positions().copy_to_host(ref_pos.data());
    atoms.get_positions().copy_to_host(pos.data());

    for (int n = 0; n < natoms; n++) {
      double dx = pos[n] - ref_pos[n];
      double dy = pos[n + natoms] - ref_pos[n + natoms];
      double dz = pos[n + 2 * natoms] - ref_pos[n + 2 * natoms];
      apply_mic(ref_atoms.box, dx, dy, dz);
      pos[n] = ref_pos[n] + dx;
      pos[n + natoms] = ref_pos[n + natoms] + dy;
      pos[n + 2 * natoms] = ref_pos[n + 2 * natoms] + dz;
    }

    atoms.get_positions().copy_from_host(pos.data());
    VCWrapper* vc_image = dynamic_cast<VCWrapper*>(&image);
    if (vc_image != nullptr) vc_image->build_positions();
  }

  void matmul_3x3(const double* a, const double* b, double* c)
  {
    for (int col = 0; col < 3; col++) {
      for (int row = 0; row < 3; row++) {
        c[row + col * 3] = 0.0;
        for (int k = 0; k < 3; k++) {
          c[row + col * 3] += a[row + k * 3] * b[k + col * 3];
        }
      }
    }
  }

  void get_reference_cell_transform(
    const Box& reference_box, const Box& box, double* transform)
  {
    matmul_3x3(box.cpu_h + 9, reference_box.cpu_h, transform);
  }

  void transform_position_to_reference_cell(
    const double* transform,
    double x,
    double y,
    double z,
    double& xr,
    double& yr,
    double& zr)
  {
    xr = x * transform[0] + y * transform[1] + z * transform[2];
    yr = x * transform[3] + y * transform[4] + z * transform[5];
    zr = x * transform[6] + y * transform[7] + z * transform[8];
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
    int index1;
    double result;
    cublasIdamax(handle, size, vec, 1, &index1);
    int index0 = index1 - 1;
    printf("max index: %d, ", index0);
    cudaMemcpy(&result, vec + index0, sizeof(double), cudaMemcpyDeviceToHost);
    return abs(result);
  }

  double __attribute__((unused)) max_abs(int size, double* vec, int nsingle, bool printflag=false)
  {
    int index1;
    double result;
    cublasIdamax(handle, size, vec, 1, &index1);
    int index0 = index1 - 1;
    if (printflag){
      int local_index = index0 % nsingle;
      printf("i_fmax: %d:%d", index0/nsingle, local_index%int(nsingle/3));
      if ((local_index + 9) % nsingle < 9) {printf("(D), ");} else {printf("(R), ");}
    }
    cudaMemcpy(&result, vec + index0, sizeof(double), cudaMemcpyDeviceToHost);
    return abs(result);
  }

  bool in_list(list<int>& mylist, int i){
    list<int>::iterator it = std::find(mylist.begin(), mylist.end(), i);
    if (it != mylist.end()) return true;
    else return false;
  }

  void print_setting(const char* name, int value, int indent=0){
    printf("%*s%-*s = %d\n", indent, "", 20-indent, name, value);
  }
  void print_setting(const char* name, bool value, int indent=0){
    printf("%*s%-*s = %s\n", indent, "", 20-indent, name, value?"true":"false");
  }
  void print_setting(const char* name, double value, int indent=0){
    printf("%*s%-*s = %g\n", indent, "", 20-indent, name, value);
  }
  void __attribute__((unused)) print_setting(
    const char* name, const char* value, int indent=0){
    printf("%*s%-*s = %s\n", indent, "", 20-indent, name, value);
  }
  void print_setting(const char* name, string value, int indent=0){
    printf("%*s%-*s = %s\n", indent, "", 20-indent, name, value.data());
  }

  bool same_fixed_cell(const Atoms& atoms, const Atoms& ref_atoms, double tolerance)
  {
    if (atoms.box.pbc_x != ref_atoms.box.pbc_x ||
        atoms.box.pbc_y != ref_atoms.box.pbc_y ||
        atoms.box.pbc_z != ref_atoms.box.pbc_z) {
      return false;
    }
    for (int i = 0; i < 9; i++) {
      if (abs(atoms.box.cpu_h[i] - ref_atoms.box.cpu_h[i]) > tolerance) return false;
    }
    return true;
  }

  void match_cell_preserve_fractional(Atoms& atoms, const Atoms& ref_atoms)
  {
    const int n_atoms = atoms.get_natoms();
    vector<double> positions(n_atoms * 3);
    atoms.get_positions().copy_to_host(positions.data());

    double old_h[18];
    memcpy(old_h, atoms.box.cpu_h, 18 * sizeof(double));
    for (int i = 0; i < n_atoms; i++) {
      const double x = positions[i];
      const double y = positions[i + n_atoms];
      const double z = positions[i + 2 * n_atoms];

      const double sx = old_h[9] * x + old_h[10] * y + old_h[11] * z;
      const double sy = old_h[12] * x + old_h[13] * y + old_h[14] * z;
      const double sz = old_h[15] * x + old_h[16] * y + old_h[17] * z;

      positions[i] =
        ref_atoms.box.cpu_h[0] * sx + ref_atoms.box.cpu_h[1] * sy + ref_atoms.box.cpu_h[2] * sz;
      positions[i + n_atoms] =
        ref_atoms.box.cpu_h[3] * sx + ref_atoms.box.cpu_h[4] * sy + ref_atoms.box.cpu_h[5] * sz;
      positions[i + 2 * n_atoms] =
        ref_atoms.box.cpu_h[6] * sx + ref_atoms.box.cpu_h[7] * sy + ref_atoms.box.cpu_h[8] * sz;
    }

    atoms.get_positions().copy_from_host(positions.data());
    atoms.box.pbc_x = ref_atoms.box.pbc_x;
    atoms.box.pbc_y = ref_atoms.box.pbc_y;
    atoms.box.pbc_z = ref_atoms.box.pbc_z;
    atoms.set_box(const_cast<double*>(ref_atoms.box.cpu_h), 9);
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
    require_option_values(param, num_param, n, 1, "neb_set");
    istate_name.assign(param[n+1]);
    n++;
  } else if (strcmp(param[n], "fs_name") == 0){
    require_option_values(param, num_param, n, 1, "neb_set");
    fstate_name.assign(param[n+1]);
    n++;
  } else if (strcmp(param[n], "mid_name") == 0){
    require_option_values(param, num_param, n, 1, "neb_set");
    mid_name.assign(param[n+1]);
    n++;
  } else if (strcmp(param[n], "mid_name_list") == 0){
    int i = n + 1;
    for (; i<num_param; i++){
      if (strcmp(param[i], "mid_name_list_end") == 0) break;
      mid_name_list.push_back(string(param[i]));
    }
    if (mid_name_list.empty()) {
      PRINT_INPUT_ERROR("mid_name_list should contain at least one filename.");
    }
    n = i;
  } else if (strcmp(param[n], "traj_name") == 0){
    require_option_values(param, num_param, n, 1, "neb_set");
    traj_name.assign(param[n+1]);
    n++;
  } else if (strcmp(param[n], "interpolate") == 0){
    require_option_values(param, num_param, n, 1, "neb_set");
    if (!is_valid_int(param[n+1], &n_interpolate)) {
      PRINT_INPUT_ERROR("interpolate should be an int.");
    }
    n++;
  } else if (strcmp(param[n], "need_relax") == 0){
    need_relax = true;

  // System Settings: cell degrees of freedom, pressure, rigid-body cleanup, and image spacing.
  } else if (strcmp(param[n], "no_vc") == 0){
    variable_cell = false;
  } else if (strcmp(param[n], "match_cell_to_initial") == 0 ||
             strcmp(param[n], "use_initial_cell") == 0){
    match_cell_to_initial = true;
  } else if (strcmp(param[n], "p") == 0){
    require_option_values(param, num_param, n, 1, "neb_set");
    double press_scalar;
    if (!is_valid_real(param[n+1], &press_scalar)) {
      PRINT_INPUT_ERROR("p should be a real.");
    }
    pressure = {press_scalar};
    n++;
  } else if (strcmp(param[n], "p3") == 0){
    require_option_values(param, num_param, n, 3, "neb_set");
    pressure.resize(3);
    for (int i=0; i<3; i++){
      if (!is_valid_real(param[n+1+i], &pressure[i])) {
        PRINT_INPUT_ERROR("p3 should be 3 reals.");
      }
    }
    n += 3;
  } else if (strcmp(param[n], "p6") == 0){
    require_option_values(param, num_param, n, 6, "neb_set");
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
  } else if (strcmp(param[n], "no_remove_translation") == 0){
    remove_translation = false;
  } else if (strcmp(param[n], "no_remove_rotation") == 0){
    remove_rotation = false;
  } else if (strcmp(param[n], "find_mic") == 0){
    find_mic = true;
  } else if (strcmp(param[n], "dist_range") == 0){
    require_option_values(param, num_param, n, 2, "neb_set");
    if (!is_valid_real(param[n+1], &min_dist) ||
        !is_valid_real(param[n+2], &max_dist)) {
      PRINT_INPUT_ERROR("dist_range should be two reals.");
    }
    n+=2;
  } else if (strcmp(param[n], "dist_ncount") == 0){
    require_option_values(param, num_param, n, 1, "neb_set");
    if (!is_valid_int(param[n+1], &dist_ncount)) {
      PRINT_INPUT_ERROR("dist_ncount should be an int.");
    }
    n++;

  // NEB Method Settings: spring model, tangent choice, and special image treatment.
  } else if (strcmp(param[n], "k") == 0){
    require_option_values(param, num_param, n, 1, "neb_set");
    if (!is_valid_real(param[n+1], &k)) {
      PRINT_INPUT_ERROR("k should be a real.");
    }
    n++;
  } else if (strcmp(param[n], "energy_based_spacing") == 0 ||
             strcmp(param[n], "energy_based_k") == 0){
    energy_based_spacing = true;
  } else if (strcmp(param[n], "energy_spacing_damping") == 0 ||
             strcmp(param[n], "energy_k_damping") == 0){
    require_option_values(param, num_param, n, 1, "neb_set");
    if (!is_valid_real(param[n+1], &energy_spacing_damping)) {
      PRINT_INPUT_ERROR("energy_spacing_damping should be a real.");
    }
    if (energy_spacing_damping <= 0.0 || energy_spacing_damping > 1.0) {
      PRINT_INPUT_ERROR("energy_spacing_damping should be in (0, 1].");
    }
    n++;
  } else if (strcmp(param[n], "energy_spacing_coeff") == 0 ||
             strcmp(param[n], "energy_k_coeff") == 0){
    require_option_values(param, num_param, n, 2, "neb_set");
    if (!is_valid_real(param[n+1], &energy_spacing_strength) ||
        !is_valid_real(param[n+2], &energy_spacing_exponent)) {
      PRINT_INPUT_ERROR("energy_spacing_coeff should be two reals.");
    }
    if (energy_spacing_strength < 0.0 || energy_spacing_strength >= 1.0) {
      PRINT_INPUT_ERROR("energy_spacing_coeff strength should be in [0, 1).");
    }
    if (energy_spacing_exponent <= 0.0) {
      PRINT_INPUT_ERROR("energy_spacing_coeff exponent should be positive.");
    }
    n += 2;
  } else if (strcmp(param[n], "energy_spacing_dist_power") == 0){
    require_option_values(param, num_param, n, 1, "neb_set");
    if (!is_valid_real(param[n+1], &energy_spacing_dist_power)) {
      PRINT_INPUT_ERROR("energy_spacing_dist_power should be a real.");
    }
    if (energy_spacing_dist_power < 0.0 || energy_spacing_dist_power > 1.0) {
      PRINT_INPUT_ERROR("energy_spacing_dist_power should be in [0, 1].");
    }
    n++;
  } else if (strcmp(param[n], "tangent") == 0){
    require_option_values(param, num_param, n, 1, "neb_set");
    tangent_method_name = string(param[n+1]);
    n++;
  } else if (strcmp(param[n], "climb") == 0){
    climb = true;
  } else if (strcmp(param[n], "find_min") == 0){
    find_min = true;
  } else if (strcmp(param[n], "dyneb") == 0 ||
             strcmp(param[n], "dynamic_relaxation") == 0){
    dynamic_relaxation = true;
  } else if (strcmp(param[n], "scale_fmax") == 0){
    require_option_values(param, num_param, n, 1, "neb_set");
    if (!is_valid_real(param[n+1], &scale_fmax)) {
      PRINT_INPUT_ERROR("scale_fmax should be a real.");
    }
    if (scale_fmax < 0.0) PRINT_INPUT_ERROR("scale_fmax should >= 0.");
    n++;
  } else if (strcmp(param[n], "dyneb_energy_exponent") == 0){
    require_option_values(param, num_param, n, 1, "neb_set");
    if (!is_valid_real(param[n+1], &dyneb_energy_exponent)) {
      PRINT_INPUT_ERROR("dyneb_energy_exponent should be a real.");
    }
    if (dyneb_energy_exponent <= 0.0) {
      PRINT_INPUT_ERROR("dyneb_energy_exponent should > 0.");
    }
    n++;
  } else if (strcmp(param[n], "dyneb_peak_width") == 0){
    require_option_values(param, num_param, n, 1, "neb_set");
    if (!is_valid_real(param[n+1], &dyneb_peak_width)) {
      PRINT_INPUT_ERROR("dyneb_peak_width should be a real.");
    }
    if (dyneb_peak_width <= 0.0 || dyneb_peak_width > 1.0) {
      PRINT_INPUT_ERROR("dyneb_peak_width should be in (0, 1].");
    }
    n++;
  } else if (strcmp(param[n], "etol") == 0){
    require_option_values(param, num_param, n, 1, "neb_set");
    if (!is_valid_real(param[n+1], &etol)) {
      PRINT_INPUT_ERROR("etol should be a real.");
    }
    n++;

  // Image Number Adjustment: controls for inserting/removing images along the path.
  } else if (strcmp(param[n], "no_ina") == 0){
    image_number_adjustment = false;
  } else if (strcmp(param[n], "trim_images") == 0){
    trim_images = true;
  } else if (strcmp(param[n], "trim_similar_tol") == 0){
    require_option_values(param, num_param, n, 1, "neb_set");
    if (!is_valid_real(param[n+1], &trim_similar_tol)) {
      PRINT_INPUT_ERROR("trim_similar_tol should be a real.");
    }
    if (trim_similar_tol <= 0.0) PRINT_INPUT_ERROR("trim_similar_tol should > 0.");
    n++;
  } else if (strcmp(param[n], "trim_etol") == 0){
    require_option_values(param, num_param, n, 1, "neb_set");
    if (!is_valid_real(param[n+1], &trim_etol)) {
      PRINT_INPUT_ERROR("trim_etol should be a real.");
    }
    has_trim_etol = true;
    n++;
  } else if (strcmp(param[n], "ina_interval") == 0){
    require_option_values(param, num_param, n, 1, "neb_set");
    if (!is_valid_int(param[n+1], &ina_interval)) {
      PRINT_INPUT_ERROR("ina_interval should be an int.");
    }
    n++;
  } else if (strcmp(param[n], "ina_local_relax_steps") == 0){
    require_option_values(param, num_param, n, 1, "neb_set");
    if (!is_valid_int(param[n+1], &ina_local_relax_steps)) {
      PRINT_INPUT_ERROR("ina_local_relax_steps should be an int.");
    }
    if (ina_local_relax_steps < 0) {
      PRINT_INPUT_ERROR("ina_local_relax_steps should be >= 0.");
    }
    n++;
  } else if (strcmp(param[n], "ina_local_relax_neighbors") == 0){
    require_option_values(param, num_param, n, 1, "neb_set");
    if (!is_valid_int(param[n+1], &ina_local_relax_neighbors)) {
      PRINT_INPUT_ERROR("ina_local_relax_neighbors should be an int.");
    }
    if (ina_local_relax_neighbors < 0) {
      PRINT_INPUT_ERROR("ina_local_relax_neighbors should be >= 0.");
    }
    n++;
  } else if (strcmp(param[n], "ina_k") == 0){
    ina_k = true;
  } else if (strcmp(param[n], "ina_k_efficient") == 0){
    require_option_values(param, num_param, n, 1, "neb_set");
    if (!is_valid_real(param[n+1], &ina_k_efficient)) {
      PRINT_INPUT_ERROR("ina_k_efficient should be a real.");
    }
    if (ina_k_efficient <= 1.0) PRINT_INPUT_ERROR("ina_k_efficient should > 1.");
    n++;
  } else if (strcmp(param[n], "ina_insert_midpoint_weight") == 0){
    require_option_values(param, num_param, n, 1, "neb_set");
    if (!is_valid_real(param[n+1], &ina_insert_midpoint_weight)) {
      PRINT_INPUT_ERROR("ina_insert_midpoint_weight should be a real.");
    }
    if (ina_insert_midpoint_weight < 0.0 || ina_insert_midpoint_weight > 1.0) {
      PRINT_INPUT_ERROR("ina_insert_midpoint_weight should be in [0, 1].");
    }
    n++;
  } else if (strcmp(param[n], "ina_force_tol") == 0){
    ina_force_tol_stages.clear();
    int previous_stage = 0;
    int i = n + 1;
    for (; i < num_param; i += 2) {
      if (strcmp(param[i], "ina_force_tol_end") == 0) break;
      if (i + 1 >= num_param || strcmp(param[i + 1], "ina_force_tol_end") == 0) {
        PRINT_INPUT_ERROR("ina_force_tol should be: stage1 tol1 [stage2 tol2 ...] [ina_force_tol_end].");
      }
      int stage;
      double residual;
      if (!is_valid_int(param[i], &stage)) {
        PRINT_INPUT_ERROR("ina_force_tol stage should be an int.");
      }
      if (stage <= previous_stage) {
        PRINT_INPUT_ERROR("ina_force_tol stages should be positive and strictly increasing.");
      }
      if (!is_valid_real(param[i + 1], &residual)) {
        PRINT_INPUT_ERROR("ina_force_tol residual should be a real.");
      }
      if (residual <= 0.0) PRINT_INPUT_ERROR("ina_force_tol residual should > 0.");
      ina_force_tol_stages.push_back({stage, residual});
      previous_stage = stage;
    }
    if (ina_force_tol_stages.empty()) {
      PRINT_INPUT_ERROR("ina_force_tol should contain at least one stage/residual pair.");
    }
    n = i;
  } else if (strcmp(param[n], "ina_check_coord") == 0){
    require_option_values(param, num_param, n, 1, "neb_set");
    if (!is_valid_int(param[n+1], &ina_check_coord)) {
      PRINT_INPUT_ERROR("ina_check_coord should be an int.");
    }
    n++;
  } else if (strcmp(param[n], "inacc_num") == 0){
    require_option_values(param, num_param, n, 1, "neb_set");
    if (!is_valid_real(param[n+1], &inacc_num)) {
      PRINT_INPUT_ERROR("inacc_num should be a real.");
    }
    if (inacc_num < 0) PRINT_INPUT_ERROR("inacc_num should >= 0");
    n++;
  } else if (strcmp(param[n], "inacc_rc") == 0){
    require_option_values(param, num_param, n, 1, "neb_set");
    if (!is_valid_real(param[n+1], &inacc_rc)) {
      PRINT_INPUT_ERROR("inacc_rc should be a real.");
    }
    if (inacc_rc <= 0) PRINT_INPUT_ERROR("inacc_rc should > 0");
    n++;
  } else if (strcmp(param[n], "cell_factor") == 0){
    require_option_values(param, num_param, n, 1, "neb_set");
    if (!is_valid_real(param[n+1], &cell_factor)) {
      PRINT_INPUT_ERROR("cell_factor should be a real.");
    }
    if (cell_factor <= 0.0) PRINT_INPUT_ERROR("cell_factor should > 0.");
    n++;
  } else if (strcmp(param[n], "active_atom_threshold") == 0){
    require_option_values(param, num_param, n, 1, "neb_set");
    if (!is_valid_real(param[n+1], &cell_metric_active_threshold)) {
      PRINT_INPUT_ERROR("active_atom_threshold should be a real.");
    }
    if (cell_metric_active_threshold <= 0.0) {
      PRINT_INPUT_ERROR("active_atom_threshold should > 0.");
    }
    n++;

  // Output Settings: trajectory/energy snapshots and progress reporting.
  } else if (strcmp(param[n], "peek_interval") == 0){
    require_option_values(param, num_param, n, 1, "neb_set");
    if (!is_valid_int(param[n+1], &peek_interval)) {
      PRINT_INPUT_ERROR("peek_interval should be an int.");
    }
    if (peek_interval <= 0) PRINT_INPUT_ERROR("peek_interval should > 0.");
    n++;
  } else if (strcmp(param[n], "dump_interval") == 0){
    require_option_values(param, num_param, n, 1, "neb_set");
    if (!is_valid_int(param[n+1], &dump_interval)) {
      PRINT_INPUT_ERROR("dump_interval should be an int.");
    }
    if (dump_interval <= 0) PRINT_INPUT_ERROR("dump_interval should > 0.");
    n++;
  } else if (strcmp(param[n], "print_interval") == 0){
    require_option_values(param, num_param, n, 1, "neb_set");
    if (!is_valid_int(param[n+1], &print_interval)) {
      PRINT_INPUT_ERROR("print_interval should be an int.");
    }
    if (print_interval <= 0) PRINT_INPUT_ERROR("print_interval should > 0.");
    n++;
  } else if (strcmp(param[n], "diagnostic_interval") == 0){
    require_option_values(param, num_param, n, 1, "neb_set");
    if (!is_valid_int(param[n+1], &diagnostic_interval)) {
      PRINT_INPUT_ERROR("diagnostic_interval should be an int.");
    }
    if (diagnostic_interval < 0) {
      PRINT_INPUT_ERROR("diagnostic_interval should >= 0.");
    }
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
    if (num_param < 2) {
      PRINT_INPUT_ERROR("neb_run should specify an optimizer.");
    }
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
    } else {
      string text = "Invalid optimizer for neb_run: ";
      text += param[1];
      PRINT_INPUT_ERROR(text.data());
    }
  } else if (strcmp(param[0], "neb_set") == 0){
    for (int n=1; n<num_param; n++){
      parse_options(param, num_param, n);
    }
  }
  
}

void NEB::reset_minimizer(
  int number_of_atoms, int max_steps, double force_tolerance, bool print_flag) {
  switch (minimizer_type) {
  case 1: {
    const char* indent = ina_local_relax_remaining > 0 ? "    " : "";
    printf("%s----------------------------------------\n", indent);
    printf("%sNew minimization, maximally %d steps.\n", indent, max_steps);

    minimizer.reset(new Minimizer_FIRE_JQH(number_of_atoms, max_steps, force_tolerance));
    auto& fire = dynamic_cast<Minimizer_FIRE_JQH&>(*minimizer);
    fire.set_cell_metric_scale(cell_metric_scale_default);
    fire.parse_FIRE(optimizer_opt.data(), optimizer_opt.size(), 0, print_flag);
    minimizer_cell_metric_scale = fire.get_cell_metric_scale();
    break;
  }
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
        Atoms* p_tmp = new Atoms(input, read_success, false);
        if (read_success) {
          // mid_list.push_back(make_pair(-1, p_tmp));
          images.emplace_back(unique_ptr<Atoms>(p_tmp));
        } else break;
      }
    }
    else {
      vector<Atoms*> raw_images;
      while (true) {
        Atoms* p_tmp = new Atoms(input, read_success, false);
        if (read_success) {
          raw_images.push_back(p_tmp);
        } else {
          delete p_tmp;
          break;
        }
      }
      if (raw_images.empty()) {
        printf("read traj failed\n");
        exit(-1);
      }
      h_ref.assign(raw_images.front()->box.cpu_h, raw_images.front()->box.cpu_h+9);
      if (cell_factor <= 0.0) {
        cell_factor = estimate_active_atom_scale(*raw_images.front(), *raw_images.back()) *
          pow(raw_images.front()->box.get_volume() / raw_images.front()->get_natoms(), 1.0 / 3.0);
      }
      for (int i = 0; i < raw_images.size(); i++) {
        if (i == 0) {
          images.push_back(make_cell_filter(*raw_images[i], pressure, remove_rotation, cell_factor));
        } else {
          images.push_back(make_cell_filter(*raw_images[i], pressure, h_ref.data(), remove_rotation, cell_factor));
          // mid_list.push_back(make_pair(-1, raw_images[i]));
        }
      }
    }
    imid_list.push_back(-1);
    printf("traj nimages: %d\n", int(images.size()));
    input.close();
  }
  else {
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
    if (!mid_name.empty() && mid_name_list.empty()) {
      mid_name_list.push_back(mid_name);
    }
    if (!mid_name_list.empty()) {
      printf("intermediate states:");
      for (const auto& name : mid_name_list) printf(" %s", name.data());
      printf("\n");
    }
    // print_arr(h_ref.data(), 9, "vector h_ref");
    if (!variable_cell){
      images.push_back(unique_ptr<Atoms>(p_is));
      for (int i=0; i<mid_name_list.size(); i++){
        Atoms *p_tmp = new Atoms((mid_name_list[i]).data());
        images.push_back(unique_ptr<Atoms>(p_tmp));
        imid_list.push_back((i+1)*n_interpolate/(mid_name_list.size()+1));
        // mid_list.push_back(make_pair((i+1)*n_interpolate/(mid_name_list.size()+1) + 1, p_tmp));
      }
      images.push_back(unique_ptr<Atoms>(p_fs));
    } else{
      if (cell_factor <= 0.0) {
        cell_factor = estimate_active_atom_scale(*p_is, *p_fs) *
          pow(p_is->box.get_volume() / p_is->get_natoms(), 1.0 / 3.0);
      }
      images.push_back(make_cell_filter(*p_is, pressure, h_ref.data(), remove_rotation, cell_factor));
      for (int i=0; i<mid_name_list.size(); i++){
        Atoms *p_tmp = new_cell_filter(
          (mid_name_list[i]).data(), pressure, h_ref.data(), remove_rotation, cell_factor);
        images.push_back(unique_ptr<Atoms>(p_tmp));
        imid_list.push_back((i+1)*n_interpolate/(mid_name_list.size()+1));
        // mid_list.push_back(make_pair((i+1)*n_interpolate/(mid_name_list.size()+1) + 1, p_tmp));
      }
      images.push_back(make_cell_filter(*p_fs, pressure, h_ref.data(), remove_rotation, cell_factor));
    }
  }
  natoms_per_image = images[0]->get_natoms();
  n_realatoms = images[0]->get_p_atoms()->get_natoms();
}

void NEB::prepare_fixed_cell_images()
{
  if (variable_cell || images.empty()) return;

  const double tolerance = 1.0e-4;
  Atoms& ref_atoms = *images.front()->get_p_atoms();
  bool all_match = true;
  for (int i = 1; i < images.size(); i++) {
    Atoms& atoms = *images[i]->get_p_atoms();
    if (!same_fixed_cell(atoms, ref_atoms, tolerance)) {
      all_match = false;
      printf(
        "no_vc cell mismatch: image %d does not use the initial cell.\n",
        i);
      printf("    initial h[0:9]:");
      for (int j = 0; j < 9; j++) printf(" %.10e", ref_atoms.box.cpu_h[j]);
      printf("\n");
      printf("    image   h[0:9]:");
      for (int j = 0; j < 9; j++) printf(" %.10e", atoms.box.cpu_h[j]);
      printf("\n");
      if (match_cell_to_initial) {
        match_cell_preserve_fractional(atoms, ref_atoms);
        printf(
          "    remapped image %d to the initial cell while preserving fractional coordinates.\n",
          i);
      }
    }
  }

  if (!all_match && !match_cell_to_initial) {
    printf(
      "ERROR: no_vc requires all input images to use the same cell. "
      "Add 'match_cell_to_initial' or 'use_initial_cell' to remap them to the initial cell.\n");
    exit(1);
  }
}

void NEB::align_images_by_mic()
{
  if (images.size() < 2) return;
  printf("Align NEB images with minimum image convention.\n");
  for (int i = 1; i < images.size(); i++) {
    align_image_by_mic(*images[i - 1], *images[i]);
  }
}

double NEB::estimate_active_atom_scale(Atoms& initial_atoms, Atoms& final_atoms)
{
  const int number_of_atoms = initial_atoms.get_natoms();
  if (number_of_atoms <= 0) return 1.0;
  if (final_atoms.get_natoms() != number_of_atoms) {
    PRINT_INPUT_ERROR("cell active atom estimate requires the same atom count in endpoint images.");
  }

  vector<double> initial_positions(initial_atoms.get_positions().size());
  vector<double> final_positions(final_atoms.get_positions().size());
  initial_atoms.get_positions().copy_to_host(initial_positions.data());
  final_atoms.get_positions().copy_to_host(final_positions.data());

  double initial_transform[9];
  double final_transform[9];
  get_reference_cell_transform(initial_atoms.box, initial_atoms.box, initial_transform);
  get_reference_cell_transform(initial_atoms.box, final_atoms.box, final_transform);

  int n_active = 0;
  for (int n = 0; n < number_of_atoms; n++) {
    // Match the cell-filter coordinate: positions transformed back to the reference cell.
    double x_initial, y_initial, z_initial;
    double x_final, y_final, z_final;
    transform_position_to_reference_cell(
      initial_transform,
      initial_positions[n],
      initial_positions[n + number_of_atoms],
      initial_positions[n + 2 * number_of_atoms],
      x_initial,
      y_initial,
      z_initial);
    transform_position_to_reference_cell(
      final_transform,
      final_positions[n],
      final_positions[n + number_of_atoms],
      final_positions[n + 2 * number_of_atoms],
      x_final,
      y_final,
      z_final);
    double dx = x_final - x_initial;
    double dy = y_final - y_initial;
    double dz = z_final - z_initial;
    if (find_mic) apply_mic(initial_atoms.box, dx, dy, dz);
    const double dr = sqrt(dx * dx + dy * dy + dz * dz);
    if (dr > cell_metric_active_threshold) n_active++;
  }

  cell_metric_active_atoms = n_active;
  return sqrt(double(max(n_active, 1)));
}

double NEB::estimate_cell_metric_scale()
{
  if (!variable_cell || n_realatoms <= 0) return 1.0;
  return estimate_active_atom_scale(*images.front()->get_p_atoms(), *images.back()->get_p_atoms());
}

void NEB::run_neb() {
  printf("\n**************************************************************\n");
  printf("*                            [o]                             *\n");
  printf("*                           /   \\                            *\n");
  printf("*                  o---o---o     o---o---o                   *\n");
  printf("*                         GPU-CFNEB                          *\n");
  printf("*                  ENTERING NEB CALCULATION                  *\n");
  printf("**************************************************************\n\n");

  initialize_images();
  prepare_fixed_cell_images();
  cell_metric_scale_default = estimate_cell_metric_scale();
  tangentmethod = get_tangent_method(tangent_method_name, k);
  dist_ncount = (dist_ncount < n_realatoms) ? dist_ncount : n_realatoms;
  if (dump_interval == -1) dump_interval = (max_steps - 1) / 10 + 1;
  if (peek_interval == -1) peek_interval = (max_steps - 1) / 50 + 1;
  if (ina_force_tol_stages.empty()) {
    ina_force_tol_stages.push_back({ina_interval, 1.0});
    ina_force_tol_stages.push_back({3 * ina_interval, 3.0});
    ina_force_tol_stages.push_back({10 * ina_interval, 1.0e100});
  }

  printf("-----------------neb settings-----------------\n");
  print_setting("k", k);
  print_setting("variable_cell", variable_cell);
  if (variable_cell){
    printf("    %-16s =", "pressure");
    for (auto x:pressure) printf(" %.4f", x);
    printf("\n");
  } else {
    print_setting("match_cell_to_initial", match_cell_to_initial, 4);
  }
  print_setting("climb", climb);
  print_setting("find_min", find_min);
  print_setting("dynamic_relaxation", dynamic_relaxation);
  if (dynamic_relaxation) {
    print_setting("scale_fmax", scale_fmax, 4);
    print_setting("dyneb_energy_exponent", dyneb_energy_exponent, 4);
    print_setting("dyneb_peak_width", dyneb_peak_width, 4);
  }
  if (climb) print_setting("etol", etol, 4);
  print_setting("image_number_adjustment", image_number_adjustment);
  if (image_number_adjustment) {
    print_setting("ina_interval", ina_interval, 4);
    print_setting("ina_local_relax_steps", ina_local_relax_steps, 4);
    if (ina_local_relax_steps > 0) {
      print_setting("ina_local_relax_neighbors", ina_local_relax_neighbors, 8);
    }
    print_setting("min_dist", min_dist, 4);
    print_setting("max_dist", max_dist, 4);
    print_setting("ina_k_efficient", ina_k_efficient, 4);
    print_setting("ina_insert_midpoint_weight", ina_insert_midpoint_weight, 4);
    printf("    %-16s =", "ina_force_tol");
    for (auto stage:ina_force_tol_stages) {
      printf(" %d %g", stage.first, stage.second);
    }
    printf("\n");
    print_setting("dist_ncount", dist_ncount, 4);
    print_setting("trim_images", trim_images, 4);
    if (trim_images) {
      print_setting("trim_similar_tol", trim_similar_tol, 8);
      print_setting("trim_etol", has_trim_etol ? trim_etol : etol, 8);
    }
    print_setting("ina_check_coord", ina_check_coord, 4);
    if (ina_check_coord) {
      print_setting("inacc_num", inacc_num, 8);
      print_setting("inacc_rc", inacc_rc, 8);
    }
  }
  print_setting("energy_based_spacing", energy_based_spacing);
  if (energy_based_spacing) {
    print_setting("energy_spacing_damping", energy_spacing_damping, 4);
    print_setting("energy_spacing_strength", energy_spacing_strength, 4);
    print_setting("energy_spacing_exponent", energy_spacing_exponent, 4);
    print_setting("energy_spacing_dist_power", energy_spacing_dist_power, 4);
  }
  print_setting("intermediate_states", !mid_name_list.empty());
  if (!mid_name_list.empty()) print_setting("n_interpolate", n_interpolate, 4);
  print_setting("need_relax", need_relax);
  print_setting("remove_translation", remove_translation);
  print_setting("remove_rotation", remove_rotation);
  print_setting("find_mic", find_mic);
  if (variable_cell) {
    print_setting(
      "cell_filter", remove_rotation ? "rotation_free" : "deformation_gradient", 4);
    VCWrapper* vc_image = dynamic_cast<VCWrapper*>(images.front().get());
    print_setting("cell_metric_active_atoms", cell_metric_active_atoms, 4);
    print_setting("active_atom_threshold", cell_metric_active_threshold, 4);
    if (vc_image != nullptr) print_setting("cell_factor", vc_image->cell_factor, 4);
  }
  print_setting("tangent_method", tangent_method_name);
  print_setting("max_steps", max_steps);
  print_setting("dump_interval", dump_interval);
  print_setting("peek_interval", peek_interval);
  print_setting("print_interval", print_interval);
  print_setting("diagnostic_interval", diagnostic_interval);
  printf("----------------------------------------------\n");

  if (diagnostic_interval > 0) {
    FILE* force_file = fopen("neb_force_components.out", "w");
    if (force_file == nullptr) {
      PRINT_INPUT_ERROR("Failed to open neb_force_components.out.");
    }
    fprintf(
      force_file,
      "# step image energy pes_perp_l2 spring_parallel_l2 dneb_l2 "
      "atom_fmax cell_fmax tangential_pes_force\n");
    fclose(force_file);

    FILE* fire_file = fopen("neb_fire_diagnostics.out", "w");
    if (fire_file == nullptr) {
      PRINT_INPUT_ERROR("Failed to open neb_fire_diagnostics.out.");
    }
    fprintf(fire_file, "# step dt power alpha n_positive reset\n");
    fclose(fire_file);

    FILE* imagewise_fire_file = fopen("neb_fire_imagewise.out", "w");
    if (imagewise_fire_file == nullptr) {
      PRINT_INPUT_ERROR("Failed to open neb_fire_imagewise.out.");
    }
    fprintf(
      imagewise_fire_file,
      "# step image dt power alpha n_positive reset\n");
    fclose(imagewise_fire_file);
  }

  if (inacc_num < 1) inacc_num *= n_realatoms;
  if (etol < 0) etol *= -n_realatoms;
  if (has_trim_etol && trim_etol < 0) trim_etol *= -n_realatoms;
  if (!dynamic_relaxation && scale_fmax != 0.0) {
    PRINT_INPUT_ERROR("scale_fmax requires dyneb (dynamic_relaxation).");
  }
  if (find_mic) align_images_by_mic();
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
    reset_minimizer(natoms_per_image, 10000, relax_tol, true);
    minimizer->compute(*images.front());
    reset_minimizer(natoms_per_image, 10000, relax_tol, true);
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
  klist.resize(images.size() - 1, k);
  energy_spacing_factor.resize(klist.size(), 1.0);
  if (images.size() <= 2 && image_number_adjustment) {
    adjust_image_spacing(false, true);
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

  double fnrm2; // used to check if minimization is finished or nimages changes
  // -------------------------main loop------------------------------
  if (count_force_calc){
    printf("INA info: step, nimages, n_force_calc, fmax\n");
    if (ina_local_relax_steps > 0) {
      printf("    INA local info: local_step, step, nimages, n_force_calc, fmax\n");
    }
  }
  while (true){
    initialize_compute();
    const int minimizer_steps =
      ina_local_relax_remaining > 0
      ? ina_local_relax_remaining
      : max_steps - step;
    reset_minimizer(natoms, minimizer_steps, force_tolerance);
    minimizer->compute(*this);
    // printf("neb total steps: %d\n", step);
    if (ina_count != 0) write_energies();
    if (step >= max_steps && ina_local_relax_remaining == 0) break;
    cublasDnrm2(handle, forces.size(), forces.data(), 1, &fnrm2);
    if (fnrm2 != 0.0) {
      // minimizer->reset_number_of_atoms((images.size()-2) * natoms_per_image);
      break;
    }
  }
  printf(
    "NEB step summary: regular=%d, INA local=%d, total=%d.\n",
    step,
    ina_local_steps_completed,
    step + ina_local_steps_completed);
  write_energies();
  write_neb_traj("final_traj.xyz", "w");
}


void NEB::compute()
{
  // printf("neb compute\n");
  // compute original forces
  set_positions();
  int force_calculations = 0;
  for (int i=1; i < nimages - 1; i++){
    // printf("image %d\n", i);
    const bool local_relaxation = ina_local_relax_remaining > 0;
    const bool local_active =
      ina_local_active.size() == nimages && ina_local_active[i];
    if ((local_relaxation && local_active) ||
        (!local_relaxation && (!dynamic_relaxation || dyneb_active[i-1]))) {
      images[i]->compute();
      force_calculations++;
    }
    images[i]->get_forces().copy_to_device(
      &forces[(i-1) * natoms_per_image*3],
      natoms_per_image*3);
    // image_energies[i] = sum(images[i]->get_potential_per_atom());
    image_energies[i] = images[i]->get_energy();
  }
  n_force_calc += force_calculations;

  find_min_max(etol);
  k_effective_list = klist;
  // printf("klist: ");
  if (energy_based_spacing) {
    if (energy_spacing_factor.size() != klist.size()) {
      energy_spacing_factor.assign(klist.size(), 1.0);
    }
    auto energy_bounds = minmax_element(image_energies.begin(), image_energies.end());
    double energy_min = *energy_bounds.first;
    double energy_max = *energy_bounds.second;
    double energy_range = energy_max - energy_min;
    for (int i=0; i<nimages-1;i++){
      double target_factor = 1.0;
      if (energy_range > 0.0) {
        double spring_energy = 0.5 * (image_energies[i] + image_energies[i+1]);
        double relative_energy = (spring_energy - energy_min) / energy_range;
        relative_energy = max(0.0, min(1.0, relative_energy));
        double energy_weight =
          (relative_energy > 0.0) ? pow(relative_energy, energy_spacing_exponent) : 0.0;
        target_factor = 1.0 - energy_spacing_strength * (1.0 - energy_weight);
      }
      double log_factor =
        (1.0 - energy_spacing_damping) * log(energy_spacing_factor[i]) +
        energy_spacing_damping * log(target_factor);
      energy_spacing_factor[i] = exp(log_factor);
      k_effective_list[i] *= energy_spacing_factor[i];
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

  const int image_size = natoms_per_image * 3;
  const bool write_diagnostics =
    diagnostic_interval > 0 && step % diagnostic_interval == 0;
  FILE* diagnostic_file = nullptr;
  vector<double> cpu_diagnostic_force(write_diagnostics ? image_size : 0);
  if (write_diagnostics) {
    diagnostic_file = fopen("neb_force_components.out", "a");
    if (diagnostic_file == nullptr) {
      PRINT_INPUT_ERROR("Failed to open neb_force_components.out.");
    }
  }
  
  for (int i=1; i < nimages - 1; i++){
    vector_substract(t2, images[i+1]->get_positions(), images[i]->get_positions());
    Spring spring2{k_effective_list[i], image_energies[i+1] - image_energies[i], t2};
    // print_gpu(t1, "t1");
    GPU_Vector<double> tangent = tangentmethod->compute_tangent(spring1, spring2);
    // print_gpu(tangent, "t");
    double tangential_force;
    cublasDdot(handle, 3*natoms_per_image, &forces[(i-1)*natoms_per_image*3], 1,
     tangent.data(), 1, &tangential_force);

    double pes_perp_l2 = 0.0;
    double spring_parallel_l2 = 0.0;
    double dneb_l2 = 0.0;
    if (write_diagnostics) {
      GPU_Vector<double> pes_perp(image_size);
      pes_perp.copy_from_device(
        &forces[(i - 1) * image_size], image_size);
      double minus_tangential_force = -tangential_force;
      cublasDaxpy(
        handle,
        image_size,
        &minus_tangential_force,
        tangent.data(),
        1,
        pes_perp.data(),
        1);
      cublasDnrm2(handle, image_size, pes_perp.data(), 1, &pes_perp_l2);

      if (tangent_method_name == "modified") {
        GPU_Vector<double> original_spring(image_size, 0.0);
        double minus_k1 = -spring1.k;
        cublasDaxpy(
          handle,
          image_size,
          &minus_k1,
          spring1.t.data(),
          1,
          original_spring.data(),
          1);
        cublasDaxpy(
          handle,
          image_size,
          &spring2.k,
          spring2.t.data(),
          1,
          original_spring.data(),
          1);

        double parallel_spring;
        cublasDdot(
          handle,
          image_size,
          original_spring.data(),
          1,
          tangent.data(),
          1,
          &parallel_spring);
        spring_parallel_l2 = abs(parallel_spring);

        GPU_Vector<double> perpendicular_spring(image_size);
        perpendicular_spring.copy_from_device(original_spring.data(), image_size);
        double minus_parallel_spring = -parallel_spring;
        cublasDaxpy(
          handle,
          image_size,
          &minus_parallel_spring,
          tangent.data(),
          1,
          perpendicular_spring.data(),
          1);

        double perpendicular_spring_l2;
        cublasDnrm2(
          handle,
          image_size,
          perpendicular_spring.data(),
          1,
          &perpendicular_spring_l2);

        GPU_Vector<double> unit_pes_perp(image_size, 0.0);
        double inverse_pes_perp_l2 = 1.0 / (pes_perp_l2 + 1.0e-10);
        cublasDaxpy(
          handle,
          image_size,
          &inverse_pes_perp_l2,
          pes_perp.data(),
          1,
          unit_pes_perp.data(),
          1);

        double perpendicular_projection;
        cublasDdot(
          handle,
          image_size,
          perpendicular_spring.data(),
          1,
          unit_pes_perp.data(),
          1,
          &perpendicular_projection);
        GPU_Vector<double> dneb_force(image_size);
        dneb_force.copy_from_device(perpendicular_spring.data(), image_size);
        double minus_perpendicular_projection = -perpendicular_projection;
        cublasDaxpy(
          handle,
          image_size,
          &minus_perpendicular_projection,
          unit_pes_perp.data(),
          1,
          dneb_force.data(),
          1);
        const double dneb_weight =
          (2.0 / M_PI) *
          atan(
            (pes_perp_l2 * pes_perp_l2) /
            (perpendicular_spring_l2 * perpendicular_spring_l2 + 1.0e-20));
        scalar_multiply(dneb_force, dneb_weight, dneb_force);
        cublasDnrm2(handle, image_size, dneb_force.data(), 1, &dneb_l2);
      } else {
        spring_parallel_l2 =
          abs(spring2.nt * spring2.k - spring1.nt * spring1.k);
      }
    }
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

    if (write_diagnostics) {
      CHECK(cudaMemcpy(
        cpu_diagnostic_force.data(),
        forces.data() + (i - 1) * image_size,
        image_size * sizeof(double),
        cudaMemcpyDeviceToHost));
      const int atomic_components =
        variable_cell ? 3 * n_realatoms : image_size;
      double atom_fmax = 0.0;
      for (int component = 0; component < atomic_components; component++) {
        atom_fmax = max(atom_fmax, abs(cpu_diagnostic_force[component]));
      }
      double cell_fmax = 0.0;
      for (int component = atomic_components; component < image_size; component++) {
        cell_fmax = max(
          cell_fmax,
          abs(cpu_diagnostic_force[component]) / minimizer_cell_metric_scale);
      }
      fprintf(
        diagnostic_file,
        "%d %d %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n",
        step,
        i,
        image_energies[i],
        pes_perp_l2,
        spring_parallel_l2,
        dneb_l2,
        atom_fmax,
        cell_fmax,
        tangential_force);
    }
    spring1 = move(spring2);
  GPU_CHECK_KERNEL;
  }
  if (diagnostic_file != nullptr) fclose(diagnostic_file);
  apply_dynamic_relaxation();
  apply_ina_local_relaxation();
  // print_gpu(forces, "neb forces");
  // print_gpu(positions, "neb pos");
}

bool NEB::update_minimizer_force_max(double force_max)
{
  bool stop_minimizer = false;
  print_info(force_max);
  const bool local_relaxation = ina_local_relax_remaining > 0;
  if (!local_relaxation && step % dump_interval == 0 && step != 0) {
    write_neb_traj("dump_traj.xyz", "a");
  }

  if (!local_relaxation && step % peek_interval == 0 && step != 0){
    write_neb_traj("peek_traj.xyz", "w");
    write_energies();
  }

  if (ina_local_relax_remaining > 0) {
    ina_local_relax_remaining--;
    const bool locally_converged = force_max < force_tolerance;
    if (locally_converged || ina_local_relax_remaining == 0) {
      const int completed_steps =
        ina_local_relax_steps - ina_local_relax_remaining;
      printf(
        "    INA local relaxation finished after %d step(s)%s.\n",
        completed_steps,
        locally_converged ? " (locally converged)" : "");
      ina_local_relax_remaining = 0;
      ina_local_active.clear();
      dyneb_active.assign(max(0, nimages - 2), true);
      forces.fill(0);
      stop_minimizer = true;
    }
    ina_local_steps_completed++;
    return stop_minimizer;
  }

  if (image_number_adjustment) adjust_image_number();
  if (image_number_adjustment && ina_count==0){
    if (print_k) print_arr(k_effective_list.data(), k_effective_list.size(), "k_effective_list");
    forces.fill(0);
    const char* indent = ina_local_relax_remaining > 0 ? "    " : "";
    printf("%simaxes before change: ", indent);
    for_each(imaxes.begin(), imaxes.end(), [](int a){printf("%d ", a);});
    printf("\n");
    stop_minimizer = true;
  }
  step++;
  return stop_minimizer;
}

void NEB::print_info(double force_max){
  auto it_max_energy = max_element(image_energies.begin(), image_energies.end());
  cudaDeviceSynchronize();
  potential_per_atom[0] = *it_max_energy - first_energy;
  fmax = force_max;
  const bool local_relaxation = ina_local_relax_remaining > 0;
  const int local_step =
    local_relaxation
    ? ina_local_relax_steps - ina_local_relax_remaining + 1
    : 0;
  const bool print_this_step =
    local_relaxation
    ? (local_step == 1 ||
       local_step % print_interval == 0 ||
       local_step == ina_local_relax_steps)
    : step % print_interval == 0;
  if (print_this_step){
    if (local_relaxation) {
      printf(
        "    INA local step: %d/%d (NEB step: %d), ",
        local_step,
        ina_local_relax_steps,
        step);
    } else {
      printf("step: %d, ", step);
    }
    printf("emax= %f(%d), ", *it_max_energy - first_energy, int(it_max_energy-image_energies.begin()));
    printf("fmax=%f\n",fmax);
    if (count_force_calc) {
      if (local_relaxation) {
        printf(
          "    INA local info: %d\t%d\t%d\t%d\t%f\n",
          local_step,
          step,
          nimages,
          n_force_calc,
          fmax);
      } else {
        printf("INA info: %d\t%d\t%d\t%f\n", step, nimages, n_force_calc, fmax);
      }
    }
  }
}

void NEB::report_minimizer_state(
  double dt, double power, double alpha, int n_positive, bool reset)
{
  if (diagnostic_interval <= 0) return;
  const int completed_step = step - 1;
  if (completed_step < 0 || completed_step % diagnostic_interval != 0) return;

  FILE* fid = fopen("neb_fire_diagnostics.out", "a");
  if (fid == nullptr) {
    PRINT_INPUT_ERROR("Failed to open neb_fire_diagnostics.out.");
  }
  fprintf(
    fid,
    "%d %.10g %.17g %.10g %d %d\n",
    completed_step,
    dt,
    power,
    alpha,
    n_positive,
    reset ? 1 : 0);
  fclose(fid);
}

void NEB::report_imagewise_minimizer_state(
  const vector<double>& dt,
  const vector<double>& power,
  const vector<double>& alpha,
  const vector<int>& n_positive,
  const vector<int>& reset)
{
  if (diagnostic_interval <= 0) return;
  const int completed_step = step - 1;
  if (completed_step < 0 || completed_step % diagnostic_interval != 0) return;
  if (
    dt.size() != power.size() ||
    dt.size() != alpha.size() ||
    dt.size() != n_positive.size() ||
    dt.size() != reset.size()) {
    PRINT_INPUT_ERROR("Invalid imagewise FIRE diagnostic state.");
  }

  FILE* fid = fopen("neb_fire_imagewise.out", "a");
  if (fid == nullptr) {
    PRINT_INPUT_ERROR("Failed to open neb_fire_imagewise.out.");
  }
  for (int image = 0; image < dt.size(); ++image) {
    fprintf(
      fid,
      "%d %d %.10g %.17g %.10g %d %d\n",
      completed_step,
      image + 1,
      dt[image],
      power[image],
      alpha[image],
      n_positive[image],
      reset[image]);
  }
  fclose(fid);
}

bool NEB::satisfy_ina_force_tolerence() const
{
  for (auto stage:ina_force_tol_stages) {
    if (ina_count < stage.first) return false;
    if (fmax < stage.second) return true;
  }
  return false;
}

void NEB::adjust_image_spacing(bool allow_remove, bool bootstrap)
{
  if (images.size() < 2) return;

  if (klist.size() != images.size() - 1) {
    klist.resize(images.size() - 1, k);
  }
  if (!energy_spacing_factor.empty() && energy_spacing_factor.size() != klist.size()) {
    energy_spacing_factor.resize(klist.size(), 1.0);
  }

  auto renormalize_klist = [&]() {
    if (ina_k && !klist.empty()) {
      double avg_k = 0.0;
      for (int j=0; j<klist.size(); j++){
        double factor =
          (energy_based_spacing && energy_spacing_factor.size() == klist.size())
          ? energy_spacing_factor[j] : 1.0;
        avg_k += klist[j] * factor;
      }
      avg_k /= klist.size();
      if (avg_k > 0.0) {
        for (int j=0; j<klist.size(); j++){
          klist[j] = klist[j] / avg_k * k;
        }
      }
    }
  };
  auto erase_image = [&](int image_index) {
    images.erase(images.begin() + image_index);
    notify_ina_image_erased(image_index);
    size_t spring_index = 0;
    if (!klist.empty()) {
      if (image_index < klist.size()) {
        spring_index = image_index;
        klist.erase(klist.begin() + image_index);
      } else {
        spring_index = klist.size() - 1;
        klist.erase(klist.end() - 1);
      }
    }
    if (!energy_spacing_factor.empty()) {
      if (spring_index < energy_spacing_factor.size()) {
        energy_spacing_factor.erase(energy_spacing_factor.begin() + spring_index);
      } else {
        energy_spacing_factor.clear();
      }
    }
  };

  GPU_Vector<double> dpos(natoms_per_image*3), new_pos(natoms_per_image*3);
  double dist;
  int max_neighbor = 10, high_coord_atom_count;
  GPU_Vector<int> cell_count(n_realatoms), cell_count_sum(n_realatoms), cell_contents(n_realatoms);
  GPU_Vector<int> NN(n_realatoms), NL(n_realatoms * max_neighbor);
  int i_ori = 0;
  for (int i = 1; i < images.size(); i++)
  {
    double cur_min_dist(min_dist), cur_max_dist(max_dist);
    int cur_dist_ncount(dist_ncount);
    i_ori++;
    if (energy_based_spacing && energy_spacing_factor.size() == klist.size()) {
      double dist_factor = energy_spacing_factor[i-1];
      if (dist_factor > 0.0) {
        double dist_range_factor = pow(dist_factor, energy_spacing_dist_power);
        cur_min_dist /= dist_range_factor;
        cur_max_dist /= dist_range_factor;
      }
    }
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
        cur_dist_ncount *= 3;
        cur_dist_ncount = (cur_dist_ncount < n_realatoms) ? cur_dist_ncount : n_realatoms;
      }
    };

    GPU_Vector<double>& pos1 = images[i-1]->get_positions();
    GPU_Vector<double>& pos2 = images[i]->get_positions();
    vector_add(dpos, pos2, pos1, 1.0, -1.0);

    GPU_Vector<double> r2_arr(n_realatoms), h2_arr(3);
    gpu_sum_square_axis1<<<(n_realatoms - 1) / 128 + 1, 128>>>(r2_arr.data(), dpos.data(), n_realatoms, 3);
    gpu_sum_square_axis1<<<1, 3>>>(h2_arr.data(), dpos.data() + 3*n_realatoms, 3, 3);
    thrust::device_ptr<double> d_ptr = thrust::device_pointer_cast(r2_arr.data());
    thrust::sort(d_ptr, d_ptr + n_realatoms);
    double h_sum_square = (variable_cell) ? sum(h2_arr.data(), 3) : 0;
    double r_sum_square = sum(r2_arr.data()+n_realatoms - cur_dist_ncount, cur_dist_ncount);
    double r_dist = sqrt(r_sum_square/cur_dist_ncount);
    double h_dist = sqrt(h_sum_square/n_realatoms);
    dist = r_dist + h_dist;

    if (bootstrap || dist > cur_max_dist){
      double max_dist_ratio = (dist > 2.0 * cur_max_dist) ? (cur_max_dist / dist) : 0.5;
      double insert_ratio =
        max_dist_ratio + ina_insert_midpoint_weight * (0.5 - max_dist_ratio);
      if (!bootstrap && dist > 2.0 * cur_max_dist && image_energies.size() == images.size()) {
        if (image_energies[i] > image_energies[i - 1]) {
          insert_ratio = 1.0 - insert_ratio;
        }
      }
      vector_add(new_pos, pos1, dpos, 1.0, insert_ratio);
      double k_old = klist[i-1];
      double spacing_factor_old =
        (energy_spacing_factor.size() == klist.size()) ? energy_spacing_factor[i-1] : 1.0;
      if (variable_cell){
        images.insert(
          images.begin()+i,
          make_cell_filter_from_position(images[0].get(), new_pos.data(), remove_rotation));
      } else {
        images.insert(images.begin()+i, make_unique<Atoms>(images[0].get(), new_pos.data()));
      }
      notify_ina_image_inserted(i);
      klist.insert(klist.begin() + i, k_old);
      if (!energy_spacing_factor.empty()) {
        energy_spacing_factor.insert(energy_spacing_factor.begin() + i, spacing_factor_old);
      }
      if (ina_k) {
        double k_left = k_old * ina_k_efficient * 2.0 * (1.0 - insert_ratio);
        double k_right = k_old * ina_k_efficient * 2.0 * insert_ratio;
        klist[i-1] = k_left;
        klist[i] = k_right;
      }
      if (bootstrap) {
        printf("add an initial image for INA, nimages: %d, dist: %.6f(r), %.6f(h), ratio: %.6f\n",
          int(images.size()), r_dist, h_dist, insert_ratio);
      } else {
        printf("add an image: %d, nimages: %d, dist: %.6f(r), %.6f(h), ratio: %.6f\n",
          i_ori, int(images.size()), r_dist, h_dist, insert_ratio);
      }
      i+=2; //skip 2 images
      i_ori++;
      ina_count = 0;
    }else if (allow_remove && images.size() > 3 && dist < cur_min_dist && i != images.size()-1){
      double k_old = klist[i-1];
      erase_image(i);
      if (ina_k) {
        double k_new = k_old / ina_k_efficient;
        if (k_new < k / 10) k_new = k_old;
        klist[i-1] = k_new;
      }
      printf("remove an image: %d , nimages: %d\n", i_ori, int(images.size()));
      i_ori++;
      ina_count = 0;
    }
    renormalize_klist();
  }
}

void NEB::adjust_image_number() {
  // printf("adjust_image_number, natoms: %d, forces.size: %d\n", natoms, forces.size());
  fflush(stdout);
  if (!satisfy_ina_force_tolerence()){
    ina_count++;
    return;
  }
  begin_ina_local_tracking();
  bool changed = false;
  auto renormalize_klist = [&]() {
    if (ina_k && !klist.empty()) {
      double avg_k = 0.0;
      for (int j=0; j<klist.size(); j++){
        double factor =
          (energy_based_spacing && energy_spacing_factor.size() == klist.size())
          ? energy_spacing_factor[j] : 1.0;
        avg_k += klist[j] * factor;
      }
      avg_k /= klist.size();
      if (avg_k > 0.0) {
        for (int j=0; j<klist.size(); j++){
          klist[j] = klist[j] / avg_k * k;
        }
      }
    }
  };
  auto erase_image = [&](int image_index) {
    images.erase(images.begin() + image_index);
    notify_ina_image_erased(image_index);
    size_t spring_index = 0;
    if (!klist.empty()) {
      if (image_index < klist.size()) {
        spring_index = image_index;
        klist.erase(klist.begin() + image_index);
      } else {
        spring_index = klist.size() - 1;
        klist.erase(klist.end() - 1);
      }
    }
    if (!energy_spacing_factor.empty()) {
      if (spring_index < energy_spacing_factor.size()) {
        energy_spacing_factor.erase(energy_spacing_factor.begin() + spring_index);
      } else {
        energy_spacing_factor.clear();
      }
    }
    changed = true;
  };

  if (trim_images && images.size() > 3) {
    find_min_max(has_trim_etol ? trim_etol : etol);
    vector<int> minima;
    minima.reserve(imins.size() + 2);
    minima.push_back(0);
    minima.insert(minima.end(), imins.begin(), imins.end());
    minima.push_back(images.size() - 1);

    vector<int> trim_indices;
    size_t head = 0;
    while (head + 1 < minima.size()) {
      bool matched = false;
      for (size_t tail = minima.size() - 1; tail > head; --tail) {
        if (compare_image(*images[minima[head]], *images[minima[tail]],
                          n_realatoms, variable_cell, trim_similar_tol)) {
          int begin = (head == 0) ? minima[head] + 1 : minima[head];
          int end = (head == 0) ? minima[tail] : minima[tail] - 1;
          end = min(end, int(images.size()) - 2);
          for (int image_index = begin; image_index <= end; ++image_index) {
            if (image_index > 0 && image_index < int(images.size()) - 1) {
              trim_indices.push_back(image_index);
            }
          }
          head = tail;
          matched = true;
          break;
        }
      }
      if (!matched) head++;
    }

    sort(trim_indices.begin(), trim_indices.end());
    trim_indices.erase(unique(trim_indices.begin(), trim_indices.end()), trim_indices.end());
    if (!trim_indices.empty()) {
      printf("trim images:");
      for (const auto image_index : trim_indices) printf(" %d", image_index);
      printf("\n");
    }
    for (auto it = trim_indices.rbegin(); it != trim_indices.rend(); ++it) {
      if (images.size() <= 3) break;
      erase_image(*it);
    }
    if (changed) {
      renormalize_klist();
      printf("nimages after trim: %d\n", int(images.size()));
      ina_count = 0;
    }
  }

  adjust_image_spacing(true, false);
  finish_ina_local_tracking();
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
          make_cell_filter_from_position(images[0].get(), cur_pos.data(), remove_rotation));
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
  const char* indent = ina_local_relax_remaining > 0 ? "    " : "";
  printf("%snimages: %d, natoms_per_image: %d\n", indent, nimages, natoms_per_image);
  natoms = (nimages - 2) * natoms_per_image; // remove first and last images

  potential_per_atom.resize(1, Memory_Type::managed);
  image_energies.resize(nimages);
  positions.resize(natoms * 3);
  forces.resize(natoms * 3, 0);
  if (dyneb_active.size() != nimages - 2) {
    dyneb_active.assign(nimages - 2, true);
  }

  build_positions();
  image_energies.front() = first_energy;
  image_energies.back() = last_energy;

}

void NEB::apply_dynamic_relaxation()
{
  if (!dynamic_relaxation || ina_local_relax_remaining > 0) return;

  const int number_of_intermediate_images = nimages - 2;
  const int image_size = natoms_per_image * 3;
  dyneb_force_max.resize(number_of_intermediate_images);
  gpu_image_force_max<<<number_of_intermediate_images, 256>>>(
    natoms_per_image,
    n_realatoms,
    number_of_intermediate_images,
    minimizer_cell_metric_scale,
    forces.data(),
    dyneb_force_max.data());
  GPU_CHECK_KERNEL;

  vector<double> force_max(number_of_intermediate_images);
  dyneb_force_max.copy_to_host(force_max.data());
  const int saddle_image = max_element(image_energies.begin() + 1, image_energies.end() - 1) -
                           image_energies.begin();
  dyneb_position_delta.resize(image_size);

  vector<double> path_coordinate(nimages, 0.0);
  for (int image = 1; image < nimages; image++) {
    GPU_Vector<double>& current_position = images[image]->get_positions();
    GPU_Vector<double>& previous_position = images[image-1]->get_positions();
    gpu_vector_substract<<<(image_size - 1) / 128 + 1, 128>>>(
      dyneb_position_delta.data(), image_size, current_position.data(), previous_position.data());
    double segment_length;
    cublasDnrm2(handle, image_size, dyneb_position_delta.data(), 1, &segment_length);
    path_coordinate[image] = path_coordinate[image-1] + segment_length;
  }
  const double path_length = path_coordinate.back();
  if (path_length > 0.0) {
    for (double& coordinate: path_coordinate) coordinate /= path_length;
  }

  vector<int> peak_images(imaxes.begin(), imaxes.end());
  if (find(peak_images.begin(), peak_images.end(), saddle_image) == peak_images.end()) {
    peak_images.push_back(saddle_image);
  }
  const auto energy_bounds = minmax_element(image_energies.begin(), image_energies.end());
  const double energy_min = *energy_bounds.first;
  const double energy_range = *energy_bounds.second - energy_min;

  for (int image = 1; image < nimages - 1; image++) {
    double energy_weight = 1.0;
    if (energy_range > 0.0) {
      double relative_energy = (image_energies[image] - energy_min) / energy_range;
      relative_energy = max(0.0, min(1.0, relative_energy));
      energy_weight = pow(relative_energy, dyneb_energy_exponent);
    }
    double peak_weight = 0.0;
    for (const int peak_image: peak_images) {
      const double relative_path =
        (path_coordinate[image] - path_coordinate[peak_image]) / dyneb_peak_width;
      peak_weight = max(peak_weight, exp(-relative_path * relative_path));
    }
    const double strictness_weight = max(energy_weight, peak_weight);
    const double local_tolerance =
      force_tolerance * (1.0 + scale_fmax * (1.0 - strictness_weight));
    const bool active =
      image == saddle_image || force_max[image - 1] >= local_tolerance;
    dyneb_active[image - 1] = active;
    if (!active) {
      CHECK(cudaMemset(
        forces.data() + (image - 1) * image_size, 0, image_size * sizeof(double)));
    }
  }
}

void NEB::apply_ina_local_relaxation()
{
  if (ina_local_relax_remaining <= 0 || ina_local_active.size() != nimages) return;
  const int image_size = natoms_per_image * 3;
  for (int image = 1; image < nimages - 1; image++) {
    if (!ina_local_active[image]) {
      CHECK(cudaMemset(
        forces.data() + (image - 1) * image_size,
        0,
        image_size * sizeof(double)));
    }
  }
}

void NEB::begin_ina_local_tracking()
{
  ina_local_tracking = ina_local_relax_steps > 0;
  ina_local_changed = false;
  if (ina_local_tracking) {
    ina_local_active.assign(images.size(), false);
  } else {
    ina_local_active.clear();
  }
}

void NEB::mark_ina_local_region(int image_index)
{
  if (!ina_local_tracking || ina_local_active.size() != images.size()) return;
  const int first_image = max(1, image_index - ina_local_relax_neighbors);
  const int last_image =
    min(int(images.size()) - 2, image_index + ina_local_relax_neighbors);
  for (int image = first_image; image <= last_image; image++) {
    ina_local_active[image] = true;
  }
}

void NEB::notify_ina_image_inserted(int image_index)
{
  if (!ina_local_tracking) return;
  if (ina_local_active.size() + 1 == images.size()) {
    ina_local_active.insert(ina_local_active.begin() + image_index, false);
  } else {
    ina_local_active.assign(images.size(), false);
  }
  ina_local_changed = true;
  mark_ina_local_region(image_index);
}

void NEB::notify_ina_image_erased(int image_index)
{
  if (!ina_local_tracking) return;
  if (ina_local_active.size() == images.size() + 1) {
    ina_local_active.erase(ina_local_active.begin() + image_index);
  } else {
    ina_local_active.assign(images.size(), false);
  }
  ina_local_changed = true;
  mark_ina_local_region(image_index - 1);
  mark_ina_local_region(image_index);
}

void NEB::finish_ina_local_tracking()
{
  ina_local_tracking = false;
  if (!ina_local_changed || ina_local_relax_steps <= 0) {
    ina_local_active.clear();
    return;
  }

  ina_local_relax_remaining = ina_local_relax_steps;
  dyneb_active.assign(max(0, int(images.size()) - 2), true);
  printf(
    "    INA local relaxation: %d step(s), active images:",
    ina_local_relax_steps);
  for (int image = 1; image < int(images.size()) - 1; image++) {
    if (ina_local_active[image]) printf(" %d", image);
  }
  printf("\n");
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
        if (abs(image_energies[extrema.front()] - image_energies.front()) < etol) {
          extrema.erase(extrema.begin());
          continue;
        }
        if (abs(image_energies[extrema.back()] - image_energies.back()) < etol) {
          extrema.erase(extrema.end() - 1);
          continue;
        }
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
  return positions;
}

void NEB::set_positions()
{
  // printf("neb set_position\n");
  for (int i=1; i<nimages-1;i++){
    const int image_size = natoms_per_image * 3;
    const bool local_relaxation = ina_local_relax_remaining > 0;
    const bool local_active =
      ina_local_active.size() == nimages && ina_local_active[i];
    const bool update_position = local_relaxation
      ? local_active
      : (!dynamic_relaxation ||
         dyneb_active.size() != nimages - 2 ||
         dyneb_active[i-1]);
    if (update_position) {
      images[i]->get_positions().copy_from_device(
        &positions[(i-1) * image_size], image_size);
    } else {
      images[i]->get_positions().copy_to_device(
        &positions[(i-1) * image_size], image_size);
    }
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
