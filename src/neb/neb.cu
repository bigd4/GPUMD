#include "neb.cuh"
#include "force/nep3.cuh"
#include <algorithm>

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

__global__ void gpu_vector_substract(double* result, const int size, double* a, double* b)
{
  int n = blockDim.x * blockIdx.x + threadIdx.x;
  if (n < size)
    result[n] = a[n] - b[n];
}

__global__ void gpu_vdot(
  double* result, const int nl,
  double* a1, double* a2, double* a3,
  double* b1, double* b2, double* b3, double alpha=1.0)
{
  int n = blockDim.x * blockIdx.x + threadIdx.x;
  if (n < nl) result[n] = alpha * (a1[n]*b1[n] + a2[n]*b2[n] + a3[n]*b3[n]);
}

void nx3_nx3_vdot(GPU_Vector<double>& result, GPU_Vector<double>& a, GPU_Vector<double>& b,
                  int nl, double alpha=1.0)
{
  gpu_vdot<<<(nl*3 -1)/128 + 1, 128>>>(result.data(), nl, 
    a.data(), a.data()+nl, a.data()+2*nl,
    b.data(), b.data()+nl, b.data()+2*nl, alpha);
}

void nx3_nx3_vdot(double* result, double* a, double* b,
                  int nl, double alpha=1.0)
{
  gpu_vdot<<<(nl*3 -1)/128 + 1, 128>>>(result, nl, 
    a, a+nl, a+2*nl,
    b, b+nl, b+2*nl, alpha);
}


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

void n_nx3_multiply(GPU_Vector<double>& result, GPU_Vector<double>& a, GPU_Vector<double>& b,
                    int nl, double alpha=1.0)
{
  for (int i=0; i<3;i++) {
  gpu_pairwise_product<<<(nl - 1) / 128 + 1, 128>>>(
    result.data() + i*nl, a.data(), b.data() + i*nl, nl, alpha);
  }
}

void n_nx3_multiply(double* result, double* a, double* b,
                  int nl, double alpha=1.0)
{
  for (int i=0; i<3;i++) {
  gpu_pairwise_product<<<(nl - 1) / 128 + 1, 128>>>(result + i*nl, a, b + i*nl, nl, alpha);
  }
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

void scalar_multiply(GPU_Vector<double>& c, const double& a, GPU_Vector<double>& b)
{
  int size = b.size();
  gpu_multiply<<<(size - 1) / 128 + 1, 128>>>(size, a, b.data(), c.data());
}

void vector_add(GPU_Vector<double>& c, GPU_Vector<double>& a, GPU_Vector<double>& b)
{
  int size = a.size();
  gpu_vector_add<<<(size - 1) / 128 + 1, 128>>>(size, a.data(), b.data(), c.data());
}
} // namespace

void ImprovedTangentMethod::compute_tangent(
  GPU_Vector<double>& tangent,
  GPU_Vector<double>& t1,
  GPU_Vector<double>& t2,
  double de1,
  double de2)
{
  int size = t1.size();
  double nt;
  printf("de1=%f, de2=%f\n", de1, de2);
  // print_gpu(t1, "t1");
  // print_gpu(t2, "t2");
  if (de1 > 0 && de2 > 0) tangent.copy_from_device(t2.data());
  else if (de1 < 0 && de2 < 0) tangent.copy_from_device(t1.data());
  else{
    double de_max = max(abs(de1), abs(de2));
    double de_min = min(abs(de1), abs(de2));
    cublasDnrm2(handle, size, t1.data(), 1, &nt1);
    cublasDnrm2(handle, size, t2.data(), 1, &nt2);
    printf("nt1= %f, nt2= %f\n", nt1, nt2);
    tangent.fill(0.0);
    if (de2 + de1 > 0){
      scale1 = de_min / nt1;
      scale2 = de_max / nt2;
    }
    else{
      scale1 = de_max / nt1;
      scale2 = de_min / nt2;
    }
    cublasDaxpy(handle, size, &scale1, t1.data(), 1, tangent.data(), 1);
    cublasDaxpy(handle, size, &scale2, t2.data(), 1, tangent.data(), 1);
  }
  cublasDnrm2(handle, size, tangent.data(), 1, &nt);
  printf("nt= %f\n", nt);
  scalar_multiply(tangent, 1/(nt+1e-10), tangent);
  // print_gpu(tangent, "tangent");
}

void ImprovedTangentMethod::compute_image_force(
  GPU_Vector<double>& springforce,
  GPU_Vector<double>& tangential_force,
  GPU_Vector<double>& tangent
  )
{
  int size = tangent.size();
  double scalar = (nt2 - nt1)*k;
  n_nx3_multiply(springforce, tangential_force, tangent, size, -1.0);
  cublasDaxpy(handle, size, &scalar, tangent.data(), 1, springforce.data(), 1);
}

NEB::NEB(){
  // variable_cell = false;
  double pressure[] = {2.0, 2.0, 2.0};
  Atoms *p_atoms = new Atoms("neb_is.xyz");
  Atoms *p_atoms2 = new Atoms("neb_fs.xyz");
  // Atoms *p_atoms3 = new Atoms("mid_hd2ll_11-6_1_20.xyz");
  // Atoms *p_atoms4 = new Atoms("test14400.xyz");
  if (!variable_cell){
    images.push_back(p_atoms);
  }
  else{
    images.push_back(new VCWrapper(*new Atoms("is.xyz"), pressure, 3));
    images.push_back(new VCWrapper(*new Atoms("mid.xyz"), pressure, 3));
    images.push_back(new VCWrapper(*new Atoms("fs.xyz"), pressure, 3));
    // images.push_back(new VCWrapper(*p_atoms3, pressure, 3));
    // images.push_back(new VCWrapper(*p_atoms2, pressure, 3));
  }
  printf("neb default construtor finishes.\n");
  natoms_per_image = images[0]->get_natoms();
  printf("neb() images[0] natoms %d\n", images[0]->get_natoms());
}


void NEB::parse_neb(const char** param, int num_param, Force& force)
{
  p_force = &force;
  if (typeid(*force.potentials[0])==typeid(NEP3)){
      printf("nep forces\n");
      int n = natoms_per_image;
      if (variable_cell) n -= 3;
      dynamic_cast<NEP3&>(*force.potentials[0]).resize(n);
  }
  for (int i=0; i < images.size(); i++) images[i]->set_calc(*p_force);
  images.front()->compute();
  images.back()->compute();
  first_energy = images.front()->get_energy();
  last_energy = images.back()->get_energy();

  const char* para[] = {"a","1"};
  dump_position.parse(para, 2, images[0]->group);
  dump_position.preprocess();
  vector<double> cpu_positions;
  cpu_positions.resize(natoms_per_image*3);

  if (strcmp(param[1], "fire") == 0) {
    minimizer_type = 1;

    if (num_param != 5) {
      PRINT_INPUT_ERROR("minimize fire should have 2 parameters.");
    }

    if (!is_valid_real(param[2], &force_tolerance)) {
      PRINT_INPUT_ERROR("Force tolerance should be a number.");
    }

    if (!is_valid_int(param[3], &max_steps)) {
      PRINT_INPUT_ERROR("Number of steps should be an integer.");
    }

    if (!is_valid_real(param[4], &k)) {
      PRINT_INPUT_ERROR("Number of steps should be an real.");
    }
    if (max_steps <= 0) {
      PRINT_INPUT_ERROR("Number of steps should > 0.");
    }
  }
  
  unique_ptr<Minimizer> minimizer;

  switch (minimizer_type) {
    case 1:
      printf("\nStart to do neb calculation.\n");
      printf("    using the fast inertial relaxation engine (FIRE) method.\n");
      printf("    with fixed box.\n");
      printf("    with a force tolerance of %g eV/A.\n", force_tolerance);
      printf("    for maximally %d steps.\n", max_steps);

      initialize();
      tangentmethod = ImprovedTangentMethod(handle, k);

      minimizer.reset(new Minimizer_FIRE_JQH(natoms, max_steps, force_tolerance));
      minimizer->compute(*this);
      for (int i=0;i<nimages;i++){
        if (variable_cell) {
          Atoms& atoms = *static_cast<VCWrapper *>(images[i])->p_atoms;
          dump_position.process(1, atoms.box, atoms.group, atoms.cpu_atom_symbol, atoms.cpu_type,
            atoms.get_positions(), cpu_positions);
          // print_gpu(atoms.get_positions());
        }
      }
      dump_position.postprocess();
      break;
    default:
      PRINT_INPUT_ERROR("Invalid minimizer.");
      break;
  }
}

void NEB::initialize() {
  printf("neb initialize\n");
  nimages = images.size();
  printf("nimage: %d\n", nimages);
  printf("natoms_per_image: %d\n", natoms_per_image);
  natoms = (nimages - 2) * natoms_per_image; // remove first and last images

  potential_per_atom.resize(1, Memory_Type::managed);
  // print_gpu(potential_per_atom, "E");
  // CHECK(cudaMallocManaged(&image_energies, nimages * sizeof(double)));
  image_energies.resize(nimages);
  positions.resize(natoms * 3);
  forces.resize(natoms * 3);

  cublasCreate(&handle);
  build_positions();
  image_energies.front() = first_energy;
  image_energies.back() = last_energy;
  
  
  // GPU_Vector<double>  t1;
  // t1.resize(natoms_per_image*3);
  // vector_substract(t1, images[2]->get_positions(), images[0]->get_positions());
  // print_gpu(t1, "fs-is");
  // print_gpu(images[0]->get_positions(), "pos_is");
  // print_gpu(images[2]->get_positions(), "pos_fs");


  // vector<double> cpu_positions;
  // cpu_positions.resize(natoms_per_image*3);

  // {Atoms& atoms = *static_cast<VCWrapper *>(images.front())->p_atoms;
  // print_gpu(atoms.get_positions(), "atom pos_fs");
  // dump_position.process(1, atoms.box, atoms.group, atoms.cpu_atom_symbol, atoms.cpu_type,
  //   atoms.get_positions(), cpu_positions);}

  // {Atoms& atoms = *static_cast<VCWrapper *>(images.back())->p_atoms;
  // print_gpu(atoms.get_positions(), "atom pos_is");
  // dump_position.process(1, atoms.box, atoms.group, atoms.cpu_atom_symbol, atoms.cpu_type,
  //   atoms.get_positions(), cpu_positions);}
}

void NEB::find_min_max()
{
  imaxes.clear();
  for (int i=1; i<nimages-1; i++){
    if (image_energies[i] > image_energies[i-1] &&
        image_energies[i] > image_energies[i+1]){
      imaxes.push_back(i);
      }
  }
}

// void NEB::find_min_max(){
//   for (int i=1, i<niamges-1;i++)
//   image_energies[i]
// }

bool in_list(list<int>& mylist, int i){
  list<int>::iterator it = std::find(mylist.begin(), mylist.end(), i);
  if (it != mylist.end()) return true;
  else return false;
}

void NEB::compute()
{
  printf("neb compute\n");
  // compute original forces
  set_positions();
  for (int i=1; i < nimages - 1; i++){
    printf("image %d\n", i);
    images[i]->compute();
    images[i]->get_forces().copy_to_device(
      &forces[(i-1) * natoms_per_image*3],
      natoms_per_image*3);
    // image_energies[i] = sum(images[i]->get_potential_per_atom());
    image_energies[i] = images[i]->get_energy();
  }
  potential_per_atom[0] = *max_element(image_energies.begin(), image_energies.end());
  print_gpu(potential_per_atom, "Ea");
  find_min_max();
  for_each(imaxes.begin(), imaxes.end(), [](int a){printf("%d ",a );});
  printf("\nimaxes\n");

  // start to compute spring force
  GPU_Vector<double> tangent, t1, t2, spring_force;
  tangent.resize(natoms_per_image*3);
  t1.resize(natoms_per_image*3);
  t2.resize(natoms_per_image*3);
  spring_force.resize(natoms_per_image*3);
  for (int i=1; i < nimages - 1; i++){
    vector_substract(t1, images[i]->get_positions(), images[i-1]->get_positions());
    vector_substract(t2, images[i+1]->get_positions(), images[i]->get_positions());
    // print_gpu(t1, "t1");
    tangentmethod.compute_tangent(tangent, t1, t2,
     image_energies[i] - image_energies[i-1], image_energies[i+1] - image_energies[i]);
    // print_gpu(tangent, "t");
    GPU_Vector<double> tangential_force;
    tangential_force.resize(natoms_per_image);
    nx3_nx3_vdot(tangential_force, images[i]->get_forces(), tangent, natoms_per_image);
    // print_gpu(tangential_force, "tangential_force");
    // if (climb && in_list())
    if (in_list(imaxes, i)){
      n_nx3_multiply(spring_force, tangential_force, tangent, natoms_per_image);
      // print_gpu(spring_force, "spring_force");
      double tmp_num = -2.0;
      cublasDaxpy(handle, natoms_per_image*3, &tmp_num,
        spring_force.data(), 1, &forces[(i-1)*natoms_per_image*3], 1);
    }
    else{
      tangentmethod.compute_image_force(spring_force, tangential_force, tangent);
      cublasDaxpy(handle, natoms_per_image*3, (new double(1.0)),
        spring_force.data(), 1, &forces[(i-1)*natoms_per_image*3], 1);
    }


  }

  // print_gpu(positions, "neb pos");
  // print_gpu(images[1]->get_forces(), "neb image1 forces");
  // print_gpu(images[0]->get_positions(), "neb image0 pos");
  // print_gpu(images[1]->get_positions(), "neb image1 pos");
  // print_gpu(forces, "neb forces");
}

GPU_Vector<double>& NEB::build_positions()
{
  printf("neb build_position\n");
  for (int i=1; i<nimages - 1; i++){
    images[i]->get_positions().copy_to_device(
      &positions[(i-1) * natoms_per_image*3],
      natoms_per_image*3);
  }
  // print_gpu(positions,"neb positions");
  return positions;
}

void NEB::set_positions()
{
  printf("neb set_position\n");
    // print_gpu(positions, "neb positions");
  for (int i=1; i<nimages-1;i++){
    // print_gpu(images[i]->get_positions(), "set_pos images[i]->get_positions()");
    images[i]->get_positions().copy_from_device(
      &positions[(i-1) * natoms_per_image*3],
      natoms_per_image*3);
  }
}

// void test(BaseAtoms& atoms){
//   printf("-------go in test-----------\n");
//   atoms.get_positions();
//   printf("-------go out of test-----------\n");
// }


void NEB::vcneb() {
  unique_ptr<Minimizer> minimizer;
  VCWrapper& image = *static_cast<VCWrapper *>(images[0]);
  const char* para[] = {"a","1"};
  printf("parse_neb image natoms %d\n", images[0]->get_natoms());
  image.set_calc(*p_force);
  minimizer.reset(new Minimizer_FIRE_JQH(images[0]->get_natoms(), max_steps, force_tolerance));
  printf("k = %f\n", k);
  dump_position.parse(para, 2, image.group);
  dump_position.preprocess();
  printf("image addr: %p\n", &image);
  Atoms& atoms = *image.p_atoms;
  vector<double> cpu_positions;
  cpu_positions.resize(atoms.get_positions().size());
  printf("------dump_position--------\n");
  dump_position.process(1, atoms.box, atoms.group, atoms.cpu_atom_symbol, atoms.cpu_type,
      atoms.get_positions(), cpu_positions);
  minimizer->compute(image);
  printf("------dump_position2--------\n");
  dump_position.process(1, atoms.box, atoms.group, atoms.cpu_atom_symbol, atoms.cpu_type,
      atoms.get_positions(), cpu_positions);
}

void NEB::norm_neb() {
  unique_ptr<Minimizer> minimizer;
  Atoms& image = *images[0];
  const char* para[] = {"a","1"};
  image.set_calc(*p_force);
  minimizer.reset(new Minimizer_FIRE_JQH(images[0]->get_natoms(), max_steps, force_tolerance));
  printf("k = %f\n", k); 
  dump_position.parse(para, 2, image.group);
  dump_position.preprocess();
  // printf("image addr: %p\n", &image);
  minimizer->compute(image);
}