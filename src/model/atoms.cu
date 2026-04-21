#include "atoms.cuh"
#include <unistd.h> // For UNIX/Linux systems
using namespace std;



void print_arr(double* a, size_t size, const char* name){
  for (int i=0;i<size;i++){
    printf("%7.3f ", a[i]);
  }
  printf("\n----arr-----%s--------------\n", name);
}

void print_arr(int* a, size_t size, const char* name){
  for (int i=0;i<size;i++){
    printf("%d ", a[i]);
  }
  printf("\n----arr-----%s--------------\n", name);
}

void print_gpu(GPU_Vector<int>& a, const char* name){
  int size = a.size();
  int temp[size];
  a.copy_to_host(temp);
  for (int i=0;i<size;i++){
    printf("%d ", temp[i]);
  }
  printf("\n---gpu------%s--------------\n", name);
}


void print_gpu(GPU_Vector<double>& a, const char* name){
  int size = a.size();
  double temp[size];
  a.copy_to_host(temp);
  for (int i=0;i<size;i++){
    printf("%7.4f ", temp[i]);
  }
  printf("\n---gpu------%s--------------\n", name);
}

void print_gpu(double* a, int size, const char* name){
  double temp[size];
  cudaMemcpy(temp, a, size*sizeof(double), cudaMemcpyDeviceToHost);
  for (int i=0;i<size;i++){
    printf("%7.4f ", temp[i]);
  }
  printf("\n---gpu arr------%s--------------\n", name);
}

namespace
{
  cublasHandle_t handle = nullptr;

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

  __global__ void gpu_multiply(const int size, double a, double* b, double* c)
  {
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < size)
      c[n] = b[n] * a;
  }



  double sum(GPU_Vector<double>& a)
  {
    double ret;
    GPU_Vector<double> result(1);
    gpu_sum<<<1, 1024>>>(a.data(), a.size(), result.data());
    result.copy_to_host(&ret);
    return ret;
  }


  void sum2d(GPU_Vector<double>& a, double* result, int len)
  {
    // double* ret;
    int nl = a.size() / len;
    GPU_Vector<double> temp(a.size());
    GPU_Vector<double> d_result(len);
    // printf("nl %d, size() %d\n", nl, a.size());
    temp.copy_from_device(a.data());
    // printf("sum2d start\n");

    for (int i=0;i<len;i++){
      gpu_sum<<<1, 1024>>>(&temp[i * nl], nl, &d_result[i]);
    }
    d_result.copy_to_host(result);
  // printf("sum2d finish\n");
  }
    __global__ void gpu_vector_add_scalar(double* result, double* a, double alpha, const int size)
  {
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < size)
      result[n] = a[n] + alpha;
  }
    // <vec> result = <vec> a + <scalar> alpha
  void vector_add_scalar(GPU_Vector<double>& result, GPU_Vector<double>& a, double& alpha)
  {
    int size = a.size();
    gpu_vector_add_scalar<<<(size - 1) / 128 + 1, 128>>>
      (result.data(), a.data(), alpha, size);
  }

  void gpu_matmul(double* mA, double* mB, double* mC,
    int M, int N, int K, int transa=CUBLAS_OP_N, int transb=CUBLAS_OP_N,
    double alpha=1.0, double beta=0.0)
  {
    int lda = (transa != CUBLAS_OP_T)? M: K;
    int ldb = (transb != CUBLAS_OP_T)? K: N;
    cublasStatus_t stat;
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

  void matmul_3x3(double* dst, double* a, double* b, int m=3, int n=3)
  {
    memset(dst, 0, sizeof(double));
    for (int i=0; i<m; i++){

      for (int j=0; j<n; j++){
        for (int k=0; k<3; k++){
          dst[i+m*j] += a[i+k*j] * b[k+j];
        }
      }
    }
  }

  double det_3x3(double *a)
  {
    double result;
      result = abs(
        a[0] * (a[4] * a[8] - a[5] * a[7]) +
        a[1] * (a[5] * a[6] - a[3] * a[8]) +
        a[2] * (a[3] * a[7] - a[4] * a[6]));
    return result;
  }

  __global__ void gpu_norm_axis1(double* dst, double* a, const int nl, const int ncol)
  {
    int n = blockDim.x * blockIdx.x + threadIdx.x;
    double sum = 0;
    if (n < nl)
      for (int i = 0; i < ncol; i++){
        sum += a[n + i * nl] * a[n + i * nl];
      }
      dst[n] = sqrtf(sum);
  }


  GPU_Vector<double> norm_axis1(GPU_Vector<double>& a, const int ncol)
  {
    int nl = a.size()/ncol;
    GPU_Vector<double> temp(nl);
    gpu_norm_axis1<<<(nl - 1) / 128 + 1, 128>>>(temp.data(), a.data(), nl, ncol);
    return temp;
  }

} // namespace



Atoms::Atoms() {
  #ifdef DEBUG
    printf("atoms default construtor for %p\n", this);
  #endif
}

Atoms::Atoms(const Atoms* p_atoms0, double* new_position):Atoms(*p_atoms0)
{
  #ifdef DEBUG
  printf("Atoms copy + position constructor %p\n", this);
  #endif
  positions.copy_from_device(new_position);
}

Atoms::Atoms(const Atoms& atoms0)
{
  #ifdef DEBUG
  printf("Atoms copy constructor %p\n", this);
  #endif
  natoms = atoms0.natoms;
  p_force = atoms0.p_force;
  cpu_atom_symbol = atoms0.cpu_atom_symbol;
  box = atoms0.box;
  type = atoms0.type;
  positions = atoms0.positions;
  potential_per_atom = atoms0.potential_per_atom;
  group = atoms0.group;
  forces.resize(natoms * 3, 0);
  virials.resize(natoms * 9);
  // printf("Atoms copy constructor finish\n");
}

Atoms::Atoms(
  Force& force0,
  Box& box0,
  GPU_Vector<double>& positions0,
  GPU_Vector<int>& type0,
  vector<Group>& group0,
  GPU_Vector<double>& potential_per_atom0,
  GPU_Vector<double>& forces0,
  GPU_Vector<double>& virials0)
{
  #ifdef DEBUG
  printf("Atoms from seperate info constructor %p\n", this);
  #endif
  natoms = type0.size();
  p_force = &force0;
  box = box0;
  positions = positions0;
  type = type0;
  group = group0;
  potential_per_atom = potential_per_atom0;
  forces = forces0;
  virials = virials0;
}

Atoms::Atoms(
  Force& force0,
  Box& box0,
  GPU_Vector<double>& positions0,
  vector<string> cpu_atom_symbol0,
  GPU_Vector<int>& type0,
  vector<Group>& group0,
  GPU_Vector<double>& potential_per_atom0,
  GPU_Vector<double>& forces0,
  GPU_Vector<double>& virials0)
  :Atoms(force0, box0, positions0, type0, group0, potential_per_atom0, forces0, virials0)
{
  cpu_atom_symbol0 = cpu_atom_symbol0;
}

// Atoms::Atoms(Atom& atom, vector<Group>& group0)
// {
//   natoms = atom.number_of_atoms;
//   positions.copy_from_device(atom.position_per_atom.data());
//   type.copy_from_device(atom.type.data());
//   // group = group0;
//   potential_per_atom.copy_from_device(atom.potential_per_atom.data());
//   forces.copy_from_device(atom.force_per_atom.data());
//   virials.copy_from_device(atom.virial_per_atom.data());
// }

Atoms::Atoms(const char* filename)
{
  printf("--------------file %s to atoms-------------------\n", filename);
  bool triclinic = true;
  int has_velocity;
  int number_of_types;
  Atom atom;
  ifstream input(filename);
  initialize_position(input, has_velocity, number_of_types, box, group, atom);
  input.close();
  initialize(atom);
}


Atoms::Atoms(ifstream& input, bool& success)
{
  bool triclinic = true;
  int has_velocity;
  int number_of_types;
  Atom atom;
  success = initialize_position(input, has_velocity, number_of_types, box, group, atom);
  if (success){
    printf("read one frame of the traj\n");
    initialize(atom);
  }
}

Atoms::~Atoms() {
  #ifdef DEBUG
    printf("atoms destructor for %p\n", this);
  #endif
  p_force = NULL;
}

void Atoms::initialize(Atom& atom) {
  const int N = atom.number_of_atoms;
  for (int m = 0; m < group.size(); ++m) {
    group[m].label.resize(N);
    group[m].size.resize(group[m].number);
    group[m].size_sum.resize(group[m].number);
    group[m].contents.resize(N);
    group[m].label.copy_from_host(group[m].cpu_label.data());
    group[m].size.copy_from_host(group[m].cpu_size.data());
    group[m].size_sum.copy_from_host(group[m].cpu_size_sum.data());
    group[m].contents.copy_from_host(group[m].cpu_contents.data());
  }
  natoms = N;
  cpu_atom_symbol = move(atom.cpu_atom_symbol);
  type.resize(N);
  type.copy_from_host(atom.cpu_type.data());
  positions.resize(N * 3);
  positions.copy_from_host(atom.cpu_position_per_atom.data());
  potential_per_atom.resize(N);
  forces.resize(N * 3, 0);
  virials.resize(N * 9);
  cudaDeviceSynchronize();
}

void Atoms::set_box(Box& box0)
{
  box = move(box0);
}

void Atoms::set_box(double* h0, int size) {
  // printf("len of cpu_h: %d\n", int(size));
  for (int i=0; i<size;i++) box.cpu_h[i] = h0[i];
  if (size == 18){}
  else if (size == 9) box.get_inverse();
  else {
    printf("setbox wrong\n");
    exit(-1);
  }
}

void Atoms::set_box(GPU_Vector<double>& h0, int size) {
  // printf("len of gpu_h: %d\n", int(size));
  h0.copy_to_host(box.cpu_h);
  if (size == 18){}
  else if (size == 9) box.get_inverse();
  else {
    printf("setbox wrong\n");
    exit(-1);
  }
}

void Atoms::compute()
{
  // printf("atoms compute\n");
  // print_gpu(positions, "r");
  // print_gpu(potential_per_atom, "e");
  // print_gpu(forces, "f");
  // print_gpu(virials, "v");
  GPU_Vector<double> tmp_positions=positions;
  p_force->compute(box, tmp_positions, type, group, potential_per_atom, forces, virials);
}

double Atoms::get_energy() { return sum(potential_per_atom);}


void VCWrapper::build_VCWrapper(vector<double> p, double* h_ref0)
{
  #ifdef DEBUG
  printf("-----VCWrapper from atoms constructor-----\n");
  #endif
  if (!handle) cublasCreate(&handle);
  initialize(p_atoms->natoms);
  CHECK(cudaMemcpy(h_ref, h_ref0, 9 * sizeof(double), cudaMemcpyHostToDevice));
  get_3x3_inverse(h_ref, h_ref+9);
  cell_factor = pow(det_3x3(h_ref), 1.0 / 3.0) * pow(natoms, 1.0 / 6.0);
  #ifdef DEBUG
  printf("cell factor: %f\n", cell_factor);
  // print_arr(h_ref, 18, "h_ref");
  #endif
  int l_p = p.size();
  if (l_p == 1){
    pressure[0] = pressure[4] = pressure[8] = p[0];
  }
  else if (l_p == 3){
    pressure[0] = p[0];
    pressure[4] = p[1];
    pressure[8] = p[2];
  }
  else if (l_p == 9){
    const int transpose_index[9] = {0, 3, 6, 1, 4, 7, 2, 5, 8};
    for (int i=0; i<9; i++) pressure[transpose_index[i]] = p[i];
  }
  else{
    printf("wrong pressure parameter number\n");
    exit(-1);
  }
  for (int i=0;i<9;i++) pressure[i] /= PRESSURE_UNIT_CONVERSION;
  build_positions();
  // print_gpu(positions, "vc positions");
  #ifdef DEBUG
  printf("wrapper constrcut finish\n");
  #endif
}

// atoms should be alive with this wrapper.
VCWrapper::VCWrapper(Atoms& atoms, vector<double> p, double* h_ref0)
{
  p_atoms.reset(&atoms);
  build_VCWrapper(p, h_ref0);
}

VCWrapper::VCWrapper(Atoms& atoms, vector<double> p)
{
  p_atoms.reset(&atoms);
  build_VCWrapper(p, atoms.box.cpu_h);
}

VCWrapper::VCWrapper(const char* filename, vector<double> p, double* h_ref0)
{
  p_atoms = make_unique<Atoms>(filename);
  build_VCWrapper(p, h_ref0);
}

VCWrapper::VCWrapper(ifstream& input, bool& success, vector<double> p, double* h_ref0)
{
  p_atoms = make_unique<Atoms>(input, success);
  if (success){
    build_VCWrapper(p, h_ref0);
  }
}

VCWrapper::VCWrapper(ifstream& input, bool& success, vector<double> p)
{
  p_atoms = make_unique<Atoms>(input, success);
  if (success){
    build_VCWrapper(p, p_atoms->box.cpu_h);
  }
}

VCWrapper::VCWrapper(const VCWrapper& vcatoms0, double* new_position)
{
  #ifdef DEBUG
  printf("VCWrapper copy from atoms0 constructor %p\n", this);
  #endif
  if (!handle) cublasCreate(&handle);
  natoms = vcatoms0.natoms;
  p_atoms.reset(new Atoms(*vcatoms0.p_atoms));
  cudaDeviceSynchronize();
  CHECK(cudaMallocManaged(&h_ref, 18 * sizeof(double)));
  CHECK(cudaMallocManaged(&deform, 18 * sizeof(double)));
  CHECK(cudaMallocManaged(&virial, 9 * sizeof(double)));
  cudaMemcpy(h_ref, vcatoms0.h_ref, 18 * sizeof(double), cudaMemcpyDeviceToDevice);
  // cudaMemcpy(deform, vcatoms0.deform, 18 * sizeof(double), cudaMemcpyDeviceToDevice);
  // cudaMemcpy(virial, vcatoms0.virial, 9 * sizeof(double), cudaMemcpyDeviceToDevice);
  GPU_CHECK_KERNEL;
  
  d_h = vcatoms0.d_h;
  // print_gpu(d_h, "d_h");
  cell_factor = vcatoms0.cell_factor;
  optimize_factor = vcatoms0.optimize_factor;
  // print_gpu(const_cast<GPU_Vector<double>&>(atoms0.positions), "atoms0.pos");
  pressure = vcatoms0.pressure;
  p_force = vcatoms0.p_force;
  cpu_atom_symbol = vcatoms0.cpu_atom_symbol;
  type = vcatoms0.type;
  group = vcatoms0.group;
  virials = vcatoms0.virials;
  forces.resize(natoms*3);
  positions.resize(natoms*3);
  #ifdef DEBUG
  printf("size1: %d, nl: %d\n", positions.size(), natoms);
  #endif
  positions.copy_from_device(new_position);
  // print_gpu(positions, "pos");
  set_positions();
  #ifdef DEBUG
  printf("VCWrapper copy from atoms0 constructor finish %p\n", this);
  #endif
}

VCWrapper::VCWrapper(Atoms* p_atoms0, double* new_position)
:VCWrapper(*dynamic_cast<VCWrapper*>(p_atoms0), new_position){}

VCWrapper::~VCWrapper() {
  #ifdef DEBUG
  printf("VCWrapper default desctructor\n");
  #endif
  // cublasDestroy(handle);
  cudaFree(h_ref);
  cudaFree(deform);
  cudaFree(virial);
}

void VCWrapper::initialize(int natoms0) {
  // printf("VCWrapper initial\n");
  if (!handle) cublasCreate(&handle);
  natoms = natoms0 + 3;
  cudaDeviceSynchronize();
  CHECK(cudaMallocManaged(&h_ref, 18 * sizeof(double)));
  CHECK(cudaMallocManaged(&deform, 18 * sizeof(double)));
  CHECK(cudaMallocManaged(&virial, 9 * sizeof(double)));
  d_h.resize(18);
  positions.resize(natoms*3);
  forces.resize(natoms*3);
  potential_per_atom.resize(1, Memory_Type::managed);
}

void VCWrapper::set_calc(Force& force) {
    // printf("set calc\n");
    p_atoms -> p_force = &force;
    }

void VCWrapper::compute() {
  // printf("vcwrapper compute\n");
  set_positions();
  // print_gpu(positions, "vc pos");
  p_atoms->compute();
    // xx xy xz    0 3 4
    // yx yy yz    6 1 5
    // zx zy zz    7 8 2
  double tmp[9];
  int virial_reorder[]={0,6,7,3,1,8,4,5,2};
  sum2d(p_atoms->virials, tmp, 9);
  // print_arr(tmp, 9, "tmp");
  double volume = p_atoms->box.get_volume();
  for (int i=0;i<9;i++) virial[i] = tmp[virial_reorder[i]] - volume * pressure[i];

  // print_arr(virial, 9, "virial");
  // first n*3 : forces @ D^T
  gpu_matmul(p_atoms->forces.data(), deform, forces.data(), natoms-3, 3, 3, 0, 1);
  // last 9 : D^(-T) @ virial
  gpu_matmul(&deform[9], virial, &forces[natoms*3-9], 3, 3, 3, 1, 0, 1/cell_factor/optimize_factor);
  // print_gpu(positions, "positions");
  // print_arr(p_atoms->box.cpu_h, 9, "cpu_h");
  // print_gpu(forces, "forces");
}

double VCWrapper::get_energy()
{
  double diag_press = (pressure[0] + pressure[4] + pressure[8]) / 3.0;
  GPU_Vector<double> F{18, Memory_Type::managed};

  // F = h h_ref^(-1)
  gpu_matmul(d_h.data(), h_ref + 9, F.data(), 3, 3, 3);

  double internal_energy = p_atoms->get_energy();
  return internal_energy + diag_press * p_atoms->box.get_volume(); 
}

GPU_Vector<double>& VCWrapper::get_potential_per_atom()
{
  return p_atoms->potential_per_atom;
}

GPU_Vector<double>& VCWrapper::build_positions()
{
  // printf("vcwrapper build_positions natoms: %d\n", natoms);
  d_h.copy_from_host(p_atoms -> box.cpu_h);
  // print_gpu(d_h, "d_h");
  compute_deform();
  // first n*3 are positions @ D^-1 (recording to h_ref)
  gpu_matmul(p_atoms->positions.data(), &deform[9], positions.data(), natoms-3, 3, 3);
  // last 9 are deform
  // CHECK(cudaMemcpy(&positions[natoms * 3 - 9], deform, 9*sizeof(double),
  //  cudaMemcpyDeviceToDevice));
  gpu_multiply<<<1, 9>>>(9, cell_factor/optimize_factor, deform, &positions[natoms * 3 - 9]);
  return positions;
}

void VCWrapper::set_positions() {
  // printf("vcwrapper set_positions\n");
  // CHECK(cudaMemcpy(deform, &positions[natoms * 3 - 9], 9*sizeof(double),
  //  cudaMemcpyDeviceToDevice));
  gpu_multiply<<<1, 9>>>(9, 1/cell_factor*optimize_factor, &positions[natoms * 3 - 9], deform);
  cudaDeviceSynchronize();
  GPU_CHECK_KERNEL;
  // CHECK(cudaMemcpy(deform, &positions[natoms * 3 - 9], 9*sizeof(double),
    // cudaMemcpyDeviceToDevice)); 
  get_3x3_inverse(deform, deform + 9);
  // print_arr(deform, 18, "deform");
  // print_gpu(positions, "positions");
  // print_gpu(p_atoms->positions, "p_atoms->positions");
  p_atoms->positions.copy_from_device(positions.data());
  // printf("size1: %d, size2: %d, nl: %d\n", positions.size(), p_atoms->positions.size(), natoms-3);
  // first n*3 : R = R~ @ D (recording to h_ref)
  gpu_matmul(positions.data(), deform, p_atoms->positions.data(), natoms-3, 3, 3);
  // last 9 : h = h0 @ D (recording to h_ref)
  gpu_matmul(h_ref, deform, d_h.data(), 3, 3, 3);
  GPU_CHECK_KERNEL;
  p_atoms->set_box(d_h, 9);
  // printf("vcwrapper set_positions finish\n");
}

void VCWrapper::compute_deform()
{
  // printf("vcwrapper compute_deform\n");
  // deform = h0^-1 @ h
  gpu_matmul(&h_ref[9], d_h.data(), deform, 3, 3, 3);
  cudaDeviceSynchronize();
  get_3x3_inverse(deform, deform + 9);
  // print_arr(deform, 18, "deform");
  // printf("compute_deform get_inverse finish\n");
}

void save_one_frame(
  FILE* fid_,
  const Box& box,
  double energy,
  double enthalpy,
  const std::vector<std::string>& cpu_atom_symbol,
  GPU_Vector<double>& position_per_atom,
  std::vector<double>& cpu_position_per_atom)
{
  #ifdef DEBUG
  printf("==========save one frame=============\n");
  #endif
  const int num_atoms_total = position_per_atom.size() / 3;
  char precision_str_[] = "%s %g %g %g\n";

  position_per_atom.copy_to_host(cpu_position_per_atom.data());
  fprintf(fid_, "%d\n", num_atoms_total);
  fprintf(
    fid_,
    "Lattice=\"%15.7e%15.7e%15.7e%15.7e%15.7e%15.7e%15.7e%15.7e%15.7e\" "
    "Properties=species:S:1:pos:R:3 "
    "energy=%.6f "
    "enthalpy=%.6f\n",
    box.cpu_h[0],
    box.cpu_h[3],
    box.cpu_h[6],
    box.cpu_h[1],
    box.cpu_h[4],
    box.cpu_h[7],
    box.cpu_h[2],
    box.cpu_h[5],
    box.cpu_h[8],
    energy,
    enthalpy);
  for (int n = 0; n < num_atoms_total; n++) {
    fprintf(
      fid_,
      precision_str_,
      cpu_atom_symbol[n].c_str(),
      cpu_position_per_atom[n],
      cpu_position_per_atom[n + num_atoms_total],
      cpu_position_per_atom[n + 2 * num_atoms_total]);
  }
  fflush(fid_);
}

void save_one_frame(
  FILE* fid_,
  const Box& box,
  double energy,
  double enthalpy,
  const std::vector<std::string>& cpu_atom_symbol,
  GPU_Vector<double>& position_per_atom)
{
  vector<double> cpu_position_per_atom(position_per_atom.size());
  save_one_frame(fid_, box, energy, enthalpy, cpu_atom_symbol,
    position_per_atom, cpu_position_per_atom);
}

void save_one_frame(
  FILE* fid_,
  const Box& box,
  double energy,
  double enthalpy,
  const std::vector<std::string>& cpu_atom_symbol,
  GPU_Vector<double>& position_per_atom,
  std::vector<double>& cpu_position_per_atom,
  vector<Group>& groups)
{
  #ifdef DEBUG
  printf("==========save one frame=============\n");
  #endif
  const int num_atoms_total = position_per_atom.size() / 3;
  char precision_str_[] = "%s %g %g %g\n";

  position_per_atom.copy_to_host(cpu_position_per_atom.data());
  fprintf(fid_, "%d\n", num_atoms_total);
  fprintf(
    fid_,
    "Lattice=\"%15.7e%15.7e%15.7e%15.7e%15.7e%15.7e%15.7e%15.7e%15.7e\" "
    "Properties=species:S:1:pos:R:3 "
    "energy=%.6f "
    "enthalpy=%.6f\n",
    box.cpu_h[0],
    box.cpu_h[3],
    box.cpu_h[6],
    box.cpu_h[1],
    box.cpu_h[4],
    box.cpu_h[7],
    box.cpu_h[2],
    box.cpu_h[5],
    box.cpu_h[8],
    energy,
    enthalpy);
  for (int n = 0; n < num_atoms_total; n++) {
    fprintf(
      fid_,
      precision_str_,
      cpu_atom_symbol[n].c_str(),
      cpu_position_per_atom[n],
      cpu_position_per_atom[n + num_atoms_total],
      cpu_position_per_atom[n + 2 * num_atoms_total]);
  }
  fflush(fid_);
}

void save_xyz_virials(
  const Box& box,
  const std::vector<std::string>& cpu_atom_symbol,
  GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  printf("==========save virials=============\n");
  FILE *fid_ = fopen("virials.xyz", "w");
  const int num_atoms_total = position_per_atom.size() / 3;
  vector<double> cpu_position_per_atom(position_per_atom.size());
  position_per_atom.copy_to_host(cpu_position_per_atom.data());
    // xx xy xz    0 3 4
    // yx yy yz    6 1 5
    // zx zy zz    7 8 2
  GPU_Vector<double> virial_sum(9), virials_shifted(virial_per_atom.size());
  vector<double> virial_avg(9);
  vector<double> cpu_virial_per_atom(virial_per_atom.size());
  sum2d(virial_per_atom, virial_sum.data(), 9);
  virial_sum.copy_to_host(virial_avg.data());
  for (auto& virial_i:virial_avg){
    virial_i /= num_atoms_total;
  }
  
  for (int i=0;i<9;i++){
    gpu_vector_add_scalar<<<(num_atoms_total-1)/128+1,128>>>(
      virials_shifted.data() + i*num_atoms_total,
      virial_per_atom.data() + i*num_atoms_total,
      -virial_avg[i], num_atoms_total);
  }
  virials_shifted.copy_to_host(cpu_virial_per_atom.data());

  vector<double> virials_norm(num_atoms_total);
  norm_axis1(virials_shifted, 9).copy_to_host(virials_norm.data());


  char precision_str_[] = "%s\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\n";

  fprintf(fid_, "%d\n", num_atoms_total);
  fprintf(
    fid_,
    "Lattice=\"%15.7e%15.7e%15.7e%15.7e%15.7e%15.7e%15.7e%15.7e%15.7e\" "
    "Properties=species:S:1:pos:R:3:virials:R:9:nv:R:1\n",
    box.cpu_h[0],
    box.cpu_h[3],
    box.cpu_h[6],
    box.cpu_h[1],
    box.cpu_h[4],
    box.cpu_h[7],
    box.cpu_h[2],
    box.cpu_h[5],
    box.cpu_h[8]);
  for (int n = 0; n < num_atoms_total; n++) {
    fprintf(
      fid_,
      precision_str_,
      cpu_atom_symbol[n].c_str(),
      cpu_position_per_atom[n],
      cpu_position_per_atom[n + num_atoms_total],
      cpu_position_per_atom[n + 2 * num_atoms_total],
      cpu_virial_per_atom[n],
      cpu_virial_per_atom[n + num_atoms_total],
      cpu_virial_per_atom[n + 2 * num_atoms_total],
      cpu_virial_per_atom[n + 3 * num_atoms_total],
      cpu_virial_per_atom[n + 4 * num_atoms_total],
      cpu_virial_per_atom[n + 5 * num_atoms_total],
      cpu_virial_per_atom[n + 6 * num_atoms_total],
      cpu_virial_per_atom[n + 7 * num_atoms_total],
      cpu_virial_per_atom[n + 8 * num_atoms_total],
      virials_norm[n]);
  }
  fflush(fid_);
}
