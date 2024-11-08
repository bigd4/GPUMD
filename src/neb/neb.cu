#include "neb.cuh"
#include "force/nep3.cuh"
#include <algorithm>

namespace
{
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

void vector_add(GPU_Vector<double>& result, GPU_Vector<double>& a, double& alpha)
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


void sum2d(GPU_Vector<double>& a, double* result, int len, int nla=0)
{
  int nl = (nla==0) ? a.size() / len : nla;
  GPU_Vector<double> temp(a.size());
  GPU_Vector<double> d_result(len);
  temp.copy_from_device(a.data());

  for (int i=0;i<len;i++){
    gpu_sum<<<1, 1024>>>(&temp[i * nl], nl, &d_result[i]);
  }
  d_result.copy_to_host(result);
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
  gpu_multiply<<<(size - 1) / 128 + 1, 128>>>(c.data(), a, b.data(), size);
}


double max_abs(cublasHandle_t& handle, int size, double* vec)
{
  int index;
  double result;
  cublasIdamax(handle, size, vec, 1, &index);
  printf("max index: %d, ", index);
  cudaMemcpy(&result, vec + index - 1, sizeof(double), cudaMemcpyDeviceToHost);
  return abs(result);
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
  // printf("de1=%f, de2=%f\n", de1, de2);
  // print_gpu(t1, "t1");
  // print_gpu(t2, "t2");
  cublasDnrm2(handle, size, t1.data(), 1, &nt1);
  cublasDnrm2(handle, size, t2.data(), 1, &nt2);
  // printf("nt1= %f, nt2= %f\n", nt1, nt2);
  if (de1 > 0 && de2 > 0) tangent.copy_from_device(t2.data());
  else if (de1 < 0 && de2 < 0) tangent.copy_from_device(t1.data());
  else{
    double de_max = max(abs(de1), abs(de2));
    double de_min = min(abs(de1), abs(de2));
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
  // printf("nt= %f\n", nt);
  scalar_multiply(tangent, 1/(nt+1e-10), tangent);
  // print_gpu(tangent, "tangent");
}

void ImprovedTangentMethod::add_image_force(
  int size,
  double& tangential_force,
  double* tangent,
  double * imgforce
  )
{
  // printf("tangential_force: %f, nt2-nt1: %f\n", tangential_force, nt2-nt1);
  double scalar = -tangential_force + (nt2 - nt1) * k;
  cublasDaxpy(handle, size, &scalar, tangent, 1, imgforce, 1);
}

NEB::NEB(){
  cublasCreate(&handle);
}

void NEB::parse_options(const char** param, int num_param, int& n){
  if (strcmp(param[n], "is_name") == 0){
    istate_name.assign(param[n+1]);
    n++;
  } else if (strcmp(param[n], "fs_name") == 0){
    fstate_name.assign(param[n+1]);
    n++;
  } else if (strcmp(param[n], "suffix") == 0){
    string suffix(param[n+1]);
    istate_name.assign("is_"+suffix+".xyz");
    fstate_name.assign("fs_"+suffix+".xyz");
    mid_name.assign("mid_"+suffix+".xyz");
    n++;
  } else if (strcmp(param[n], "mid_name") == 0){
    mid_name.assign(param[n+1]);
    n++;
  } else if (strcmp(param[n], "k") == 0){
    if (!is_valid_real(param[n+1], &k)) {
      PRINT_INPUT_ERROR("k should be an real.");
    }
    n++;
  } else if (strcmp(param[n], "p") == 0){
    if (!is_valid_real(param[n+1], &pressure)) {
      PRINT_INPUT_ERROR("p should be an real.");
    }
    n++;
  } else if (strcmp(param[n], "interpolate") == 0){
    if (!is_valid_int(param[n+1], &n_interpolate)) {
      PRINT_INPUT_ERROR("interpolate should be an real.");
    }
    n++;
  } else if (strcmp(param[n], "dist_range") == 0){
    if (!is_valid_real(param[n+1], &min_dist) ||
        !is_valid_real(param[n+2], &max_dist)) {
      PRINT_INPUT_ERROR("dist_range should be two reals.");
    }
    n+=2;
  } else if (strcmp(param[n], "dump_interval") == 0){
    if (!is_valid_int(param[n+1], &dump_interval)) {
      PRINT_INPUT_ERROR("dump_interval should be an int.");
    }
    if (dump_interval <= 0) PRINT_INPUT_ERROR("dump_interval should > 0.");
    n++;
  } else if (strcmp(param[n], "has_mid") == 0){
    has_mid = true;
  } else if (strcmp(param[n], "climb") == 0){
    climb = true;
  } else if (strcmp(param[n], "need_relax") == 0){
    need_relax = true;
  } else if (strcmp(param[n], "climb") == 0){
    need_relax = true;
  } else {
    string text="no keyword match with: ";
    text += param[n];
    PRINT_INPUT_ERROR(text.data());
  }
}

void NEB::parse_neb(const char** param, int num_param, Force& force)
{
  p_force = &force;

  tangentmethod = ImprovedTangentMethod(handle, k);
  
  if (strcmp(param[0], "neb_run") == 0) {
    if (strcmp(param[1], "fire") == 0) {
      minimizer_type = 1;
      if (num_param < 4) {
        PRINT_INPUT_ERROR("minimize fire should have 2 parameters.");
      }

      if (!is_valid_real(param[2], &force_tolerance)) {
        PRINT_INPUT_ERROR("Force tolerance should be a number.");
      }

      if (!is_valid_int(param[3], &max_steps)) {
        PRINT_INPUT_ERROR("Number of steps should be an integer.");
      }
      for (int n=4; n<num_param; n++){
        parse_options(param, num_param, n);
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
    break;
  default:
    PRINT_INPUT_ERROR("Invalid minimizer.");
    break;
  }
}

void NEB::run_neb() {
  // variable_cell = false;
  vector<double> press_in = {pressure};
  Atoms *p_is = new Atoms(istate_name.data());
  Atoms *p_fs = new Atoms(fstate_name.data());
  Atoms *p_mid;
  if (has_mid) p_mid = new Atoms(mid_name.data());
  ref_h.assign(p_is->box.cpu_h, p_is->box.cpu_h+9);
  print_arr(ref_h.data(), 9, "vector ref_h");
  if (!variable_cell){
    images.push_back(p_is);
    if (has_mid) images.push_back(p_mid);
    images.push_back(p_fs);
  }
  else{
    optimize_factor = pow(p_is->get_natoms(), 1.0/4);
    printf("optimize_factor=%f\n", optimize_factor);
    images.push_back(new VCWrapper(*p_is, press_in));
    if (has_mid){
      Atoms *p_tmp = new VCWrapper(*p_mid, press_in, ref_h.data());
      images.push_back(p_tmp);
      mid_list[n_interpolate/2 + 1] = p_tmp;
    }
    images.push_back(new VCWrapper(*p_fs, press_in, ref_h.data()));
    if (remove_transition){
      double center[3];
      double ref_center[3];
      ref_center[0] = (ref_h[0] + ref_h[1] + ref_h[2])/2;
      ref_center[1] = (ref_h[3] + ref_h[4] + ref_h[5])/2;
      ref_center[2] = (ref_h[6] + ref_h[7] + ref_h[8])/2;
      for (auto it=images.begin();it!=images.end();it++){
        int natoms = (*it)->get_p_atoms()->type.size();
        GPU_Vector<double>& pos = (*it)->get_positions();
        sum2d(pos, center, 3, natoms);
        // print_arr(center, 3, "center");
        for (int i=0;i<3;i++){
          center[i] /= natoms;
          gpu_vector_add_scalar<<<(natoms-1)/128+1,128>>>
              (pos.data() + i*natoms, pos.data() + i*natoms, ref_center[i]-center[i], natoms);
        }
      }
    }
  }
  natoms_per_image = images[0]->get_natoms();
  // for dump_position
  const char* para[] = {"","1"};
  dump_position.parse(para, 2, images[0]->group);
  dump_position.preprocess();
  //
  // printf("force id: %s, nep id: %s\n",typeid(*p_force->potentials[0]).name(), typeid(NEP3).name());
  // reinitialize nep to make sure that natom in it is right
  if (typeid(*(p_force->potentials[0]))==typeid(NEP3)){
    printf("nep forces\n");
    int n = natoms_per_image;
    if (variable_cell) n -= 3;
    dynamic_cast<NEP3&>(*p_force->potentials[0]).resize(n);
  }
  for (int i=0; i < images.size(); i++) images[i]->set_calc(*p_force);
  if (need_relax){
    printf("-----------relax---------\n");
    reset_minimizer(natoms_per_image, 10000, 0.001);
    minimizer->compute(*images.front());
    reset_minimizer(natoms_per_image, 10000, 0.001);
    minimizer->compute(*images.back());
    printf("-----------relax finish---------\n");
    write_neb_traj();
  }
  if (n_interpolate > 0){
    interpolate(n_interpolate);
  }
  printf("neb() images[0] natoms %d\n", images[0]->get_natoms());
  // natoms = (images.size()-2) * natoms_per_image;
  // print_arr(images[0]->get_p_atoms()->box.cpu_h, 18, "box.h");

  images.front()->compute();
  images.back()->compute();
  first_energy = images.front()->get_energy();
  last_energy = images.back()->get_energy();

  double fnrm2; // used to check if minimization is finished or nimages changes
  while (true){
    initialize_compute();
    reset_minimizer(natoms, max_steps - step, force_tolerance);
    minimizer->compute(*this);
    printf("neb total steps: %d\n", step);

    cublasDnrm2(handle, natoms_per_image*3, forces.data(), 1, &fnrm2);
    if (fnrm2 != 0.0) {
      // minimizer->reset_number_of_atoms((images.size()-2) * natoms_per_image);
      break;
    }
  }
  write_neb_traj();
  dump_position.postprocess();
}

void NEB::write_neb_traj(){
  printf("============write neb traj==============\n");
  vector<double> cpu_positions(natoms_per_image*3);
  for (int i=0;i<images.size();i++){
    Atoms& atoms = *images[i]->get_p_atoms();
    dump_position.process(1, atoms.box, atoms.group, atoms.cpu_atom_symbol, atoms.cpu_type,
      atoms.get_positions(), cpu_positions);
    // print_gpu(atoms.get_positions());
  }
}

void NEB::interpolate(int n) {
  // printf("neb interpolate, size of images[0]->get_positions().size()=%d\n", images[0]->get_positions().size());
  GPU_Vector<double> dpos(images[0]->get_positions().size()), cur_pos(images[0]->get_positions().size());
  vector<int> i_keyframe={0};
  vector<Atoms*> keyframe={images.front()};
  int n_key=0;
  for (auto it=mid_list.begin(); it!=mid_list.end();it++){
    i_keyframe.push_back(it->first+n_key);
    keyframe.push_back(it->second);
    n_key++;
  }
  i_keyframe.push_back(n_interpolate+n_key+1);
  keyframe.push_back(images.back());
  print_arr(i_keyframe.data(), i_keyframe.size(), "i_k");  
  for (int k=0; k<n_key+1;k++){
    GPU_Vector<double>& ipos = keyframe[k]->get_positions();
    GPU_Vector<double>& fpos = keyframe[k+1]->get_positions();
    vector_add(dpos, fpos, ipos, 1.0, -1.0);
    // printf("ipos len = %d, fpos len = %d\n", ipos.size(), fpos.size());
    // print_gpu(dpos, "dpos");
    int n_cur = i_keyframe[k+1] - i_keyframe[k];
    for (int i_cur=1;i_cur<=n_cur;i_cur++){
      printf("k=%d, i_cur=%d, nimages=(%d)%d\n",k,i_cur,i_keyframe[k]+k+i_cur,images.size());
      vector_add(cur_pos, ipos, dpos, 1, double(i_cur)/(n_cur+1));
      if (variable_cell){
        // print_gpu(cur_pos,"cur_pos");
        VCWrapper* new_vcatoms = new VCWrapper(images[0], cur_pos.data());
        images.insert(images.begin()+i_keyframe[k]+k+i_cur, new_vcatoms);
      } else {
        Atoms* new_atoms = new Atoms(*images[0], cur_pos.data());
        images.insert(images.begin()+i_keyframe[k]+k+i_cur, new_atoms);
      }
    }
  }
  // printf("neb interpolate finish\n");
}

void NEB::initialize_compute() {
  printf("neb initialize\n");
  nimages = images.size();
  printf("nimages: %d, natoms_per_image: %d\n", nimages, natoms_per_image);
  natoms = (nimages - 2) * natoms_per_image; // remove first and last images

  potential_per_atom.resize(1, Memory_Type::managed);
  // print_gpu(potential_per_atom, "E");
  // CHECK(cudaMallocManaged(&image_energies, nimages * sizeof(double)));
  image_energies.resize(nimages);
  positions.resize(natoms * 3);
  forces.resize(natoms * 3, 0);

  build_positions();
  image_energies.front() = first_energy;
  image_energies.back() = last_energy;
  
  // GPU_Vector<double>  t1;
  // t1.resize(natoms_per_image*3);
  // vector_substract(t1, images[2]->get_positions(), images[0]->get_positions());
  // print_gpu(t1, "fs-is");
  // print_gpu(images[0]->get_positions(), "pos_is");
  // print_gpu(images[2]->get_positions(), "pos_fs");
}

bool in_list(list<int>& mylist, int i){
  list<int>::iterator it = std::find(mylist.begin(), mylist.end(), i);
  if (it != mylist.end()) return true;
  else return false;
}

void NEB::compute()
{
  // printf("neb compute\n");
  // compute original forces
  
  for (int i=1; i < nimages - 1; i++){
    // &forces[(i-1) * natoms_per_image*3]
    gpu_multiply<<<1, 9>>>(positions.data() + i*natoms_per_image*3 - 9,
          optimize_factor, positions.data() + i*natoms_per_image*3 - 9, 9);
  }
  set_positions();
  for (int i=1; i < nimages - 1; i++){
    // printf("image %d\n", i);
    images[i]->compute();
    images[i]->get_forces().copy_to_device(
      &forces[(i-1) * natoms_per_image*3],
      natoms_per_image*3);
    // image_energies[i] = sum(images[i]->get_potential_per_atom());
    image_energies[i] = images[i]->get_energy();
  }
  
  int base = (max_steps >= 100) ? (max_steps / 100) : 1;
  if (step % base == 0 ){
    printf("image_energies: ");
    for_each(image_energies.begin(), image_energies.end(),
            [this](double i){printf("%.4f ", i - first_energy);});
    double max_energy = *max_element(image_energies.begin(), image_energies.end());
    potential_per_atom[0] = max_energy;
    printf("\nEmax=%f, Ei=%f, Ef=%f\n", max_energy, max_energy-first_energy, max_energy-last_energy);

  }
  if (dump_interval == -1){
    if (step % (10* base) == 0 ) write_neb_traj();
  } else if (step % dump_interval == 0) write_neb_traj();

  find_min_max();
  // printf("imaxes: ");
  // for_each(imaxes.begin(), imaxes.end(), [](int a){printf("%d ",a );});
  // printf("\n");

  // start to compute spring force
  GPU_Vector<double> tangent(natoms_per_image*3),
                      t1(natoms_per_image*3),
                      t2(natoms_per_image*3),
                      spring_force(natoms_per_image*3);
  for (int i=1; i < nimages - 1; i++){
    vector_substract(t1, images[i]->get_positions(), images[i-1]->get_positions());
    vector_substract(t2, images[i+1]->get_positions(), images[i]->get_positions());
    // print_gpu(t1, "t1");
    tangentmethod.compute_tangent(tangent, t1, t2,
     image_energies[i] - image_energies[i-1], image_energies[i+1] - image_energies[i]);
    // print_gpu(tangent, "t");
    double tangential_force;
    cublasDdot(handle, 3*natoms_per_image, images[i]->get_forces().data(), 1,
     tangent.data(), 1, &tangential_force);
    // print_gpu(tangential_force, "tangential_force");
    // if (climb && in_list())
    if (in_list(imaxes, i)){
      // print_gpu(spring_force, "spring_force");
      double tmp_num = -2.0 * tangential_force;
      cublasDaxpy(handle, natoms_per_image*3, &tmp_num,
        tangent.data(), 1, &forces[(i-1)*natoms_per_image*3], 1);
    }
    else{
      tangentmethod.add_image_force(natoms_per_image*3,
       tangential_force, tangent.data(), &forces[(i-1)*natoms_per_image*3]);
      // cublasDaxpy(handle, natoms_per_image*3, (new double(1.0)),
      //   spring_force.data(), 1, &forces[(i-1)*natoms_per_image*3], 1);
    }

  CUDA_CHECK_KERNEL;
  }
  check_dist();
  for (int i=1; i < nimages - 1; i++){
    // &forces[(i-1) * natoms_per_image*3]
    gpu_multiply<<<1, 9>>>(forces.data() + i*natoms_per_image*3 - 9,
          1/optimize_factor, forces.data() + i*natoms_per_image*3 - 9, 9);
    gpu_multiply<<<1, 9>>>(positions.data() + i*natoms_per_image*3 - 9,
          1/optimize_factor, positions.data() + i*natoms_per_image*3 - 9, 9);
  }
  step++;
  // print_gpu(forces, "neb forces");
  // print_gpu(positions, "neb pos");
}

void NEB::check_dist() {
  // printf("check_dist, natoms: %d, forces.size: %d\n", natoms, forces.size());
  double fmax = max_abs(handle, natoms*3, forces.data());
  printf("fmax=%f\n",fmax);
  if (vi_count < vi_interval || (vi_count < vi_interval *2 && fmax > 2) ||
      (vi_count < vi_interval *5 && fmax > 3) || fmax > 5){
    vi_count++;
    return;
  }
  GPU_Vector<double> dpos(natoms_per_image*3), new_pos(natoms_per_image*3);
  double nrm2, dist;
  // for (auto it = images.begin()+1; it != images.end()-1; it++)
  // printf("dist:");
  for (int i = 1; i < images.size(); i++)
  {
    GPU_Vector<double>& pos1 = images[i-1]->get_positions();
    GPU_Vector<double>& pos2 = images[i]->get_positions();
    vector_add(dpos, pos2, pos1, 1.0, -1.0);

    //calc_dist
    cublasDnrm2(handle, natoms_per_image*3, dpos.data(), 1, &nrm2);
    dist = nrm2/sqrt(natoms_per_image);
    // printf(" %f ", dist);
    if (dist > max_dist){
      printf("imaxes: ");
      for_each(imaxes.begin(), imaxes.end(), [](int a){printf("%d ",a );});
      printf("\n");

      vector_add(new_pos, pos1, pos2, 0.5, 0.5);
      // print_gpu(new_pos, "new_pos");
      images.insert(images.begin()+i, new VCWrapper(images[0], new_pos.data()));
      printf("add an image: %d , nimages: %d\n", i, images.size());
      i+=2; //skip 2 images
      vi_count = 0;
    }else if (dist < min_dist && i != images.size()-1){
      delete(images[i]);
      images.erase(images.begin()+i);
      printf("remove an image: %d , nimages: %d\n", i, images.size());
      // i--; // skip 2 images
      vi_count = 0;
    }
  }
  
  if (vi_count==0) initialize_compute();
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

GPU_Vector<double>& NEB::build_positions()
{
  printf("neb build_position\n");
  for (int i=1; i<nimages - 1; i++){
    images[i]->get_positions().copy_to_device(
      &positions[(i-1) * natoms_per_image*3],
      natoms_per_image*3);
  }
  for (int i=1; i < nimages - 1; i++){
    gpu_multiply<<<1, 9>>>(positions.data() + i*natoms_per_image*3 - 9,
          1/optimize_factor, positions.data() + i*natoms_per_image*3 - 9, 9);
  }
  // print_gpu(positions,"neb positions");
  return positions;
}

void NEB::set_positions()
{
  // printf("neb set_position\n");
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

// void NEB::vcneb() {
//   unique_ptr<Minimizer> minimizer;
//   VCWrapper& image = *dynamic_cast<VCWrapper *>(images[0]);
//   const char* para[] = {"a","1"};
//   printf("parse_neb image natoms %d\n", images[0]->get_natoms());
//   image.set_calc(*p_force);
//   minimizer.reset(new Minimizer_FIRE_JQH(images[0]->get_natoms(), max_steps, force_tolerance));
//   printf("k = %f\n", k);
//   dump_position.parse(para, 2, image.group);
//   dump_position.preprocess();
//   printf("image addr: %p\n", &image);
//   Atoms& atoms = *image.p_atoms;
//   vector<double> cpu_positions(atoms.get_positions().size());
//   printf("------dump_position--------\n");
//   dump_position.process(1, atoms.box, atoms.group, atoms.cpu_atom_symbol, atoms.cpu_type,
//       atoms.get_positions(), cpu_positions);
//   minimizer->compute(image);
//   printf("------dump_position2--------\n");
//   dump_position.process(1, atoms.box, atoms.group, atoms.cpu_atom_symbol, atoms.cpu_type,
//       atoms.get_positions(), cpu_positions);
// }

// void NEB::norm_neb() {
//   unique_ptr<Minimizer> minimizer;
//   Atoms& image = *images[0];
//   const char* para[] = {"a","1"};
//   image.set_calc(*p_force);
//   minimizer.reset(new Minimizer_FIRE_JQH(images[0]->get_natoms(), max_steps, force_tolerance));
//   printf("k = %f\n", k); 
//   dump_position.parse(para, 2, image.group);
//   dump_position.preprocess();
//   // printf("image addr: %p\n", &image);
//   minimizer->compute(image);
// }

void process(
  FILE* fid_,
  const Box& box,
  const std::vector<std::string>& cpu_atom_symbol,
  const std::vector<int>& cpu_type,
  double enthalpy,
  GPU_Vector<double>& position_per_atom,
  std::vector<double>& cpu_position_per_atom)
{
  const int num_atoms_total = position_per_atom.size() / 3;
  char precision_str_[] = "%s %g %g %g\n";

  position_per_atom.copy_to_host(cpu_position_per_atom.data());
  fprintf(fid_, "%d\n", num_atoms_total);
  fprintf(
    fid_,
    "Lattice=\"%15.7e%15.7e%15.7e%15.7e%15.7e%15.7e%15.7e%15.7e%15.7e\" "
    "Properties=species:S:1:pos:R:3 "
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
