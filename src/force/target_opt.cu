#include "target_opt.cuh"
#include "model/read_xyz.cuh"
#include "model/atoms.cuh"
using namespace std;


static __global__ void get_dpos_target(
  const int natoms,
  const Box box,
  const int* NN,
  const int* NL,
  const int* i_pick,
  const int n_pick,
  const double* g_x,
  const double* g_y,
  const double* g_z,
  double* __restrict__ g_dx,
  double* __restrict__ g_dy,
  double* __restrict__ g_dz)
{
  int nid = blockIdx.x * blockDim.x + threadIdx.x;
  if (nid < n_pick){
    int n1 = i_pick[nid]; // particle index
    int neighbor_number = NN[n1];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];

    for (int i2=0; i2<neighbor_number; i2++){
      int n2 = NL[n1 + i2 * natoms];
      double x12 = g_x[n2] - x1;
      double y12 = g_y[n2] - y1;
      double z12 = g_z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      g_dx[nid + i2 * n_pick] = x12;
      g_dy[nid + i2 * n_pick] = y12;
      g_dz[nid + i2 * n_pick] = z12;
    }
  }
}

// notice that virial_per_atom is not meaningful here, only total virial is guarenteed.
static __global__ void calc_spring_force(
  const int natoms,
  const Box box,
  const int* NN,
  const int* NL,
  const int* i_pick,
  const int n_pick,
  const double k,
  const double vert_part,
  const double* g_x,
  const double* g_y,
  const double* g_z,
  const double* g_dx,
  const double* g_dy,
  const double* g_dz,
  double* f_x,
  double* f_y,
  double* f_z,
  double* g_virial)
{
  int nid = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (nid < n_pick){
    // double s_fx = 0.0;                                   // force_x
    // double s_fy = 0.0;                                   // force_y
    // double s_fz = 0.0;                                   // force_z
    double s_sxx = 0.0;                                  // virial_stress_xx
    double s_sxy = 0.0;                                  // virial_stress_xy
    double s_sxz = 0.0;                                  // virial_stress_xz
    double s_syx = 0.0;                                  // virial_stress_yx
    double s_syy = 0.0;                                  // virial_stress_yy
    double s_syz = 0.0;                                  // virial_stress_yz
    double s_szx = 0.0;                                  // virial_stress_zx
    double s_szy = 0.0;                                  // virial_stress_zy
    double s_szz = 0.0;                                  // virial_stress_zz

    int n1 = i_pick[nid]; // particle index
    int neighbor_number = NN[n1];
    double x1 = g_x[n1];
    double y1 = g_y[n1];
    double z1 = g_z[n1];

    for (int i2=0; i2<neighbor_number; i2++){
      int n2 = NL[n1 + i2 * natoms];
      double x12 = g_x[n2] - x1;
      double y12 = g_y[n2] - y1;
      double z12 = g_z[n2] - z1;
      apply_mic(box, x12, y12, z12);
      double dist = sqrt(x12 * x12 + y12 * y12 + z12 * z12);
      double u_x12 = x12 / dist;
      double u_y12 = y12 / dist;
      double u_z12 = z12 / dist;

      double target_x12 = g_dx[nid + i2 * n_pick];
      double target_y12 = g_dy[nid + i2 * n_pick];
      double target_z12 = g_dz[nid + i2 * n_pick];

      double diff_x = x12 - target_x12;
      double diff_y = y12 - target_y12;
      double diff_z = z12 - target_z12;

      double diff_par_dist = (diff_x * u_x12 + diff_y * u_y12 + diff_z * u_z12);
      double diff_par_x = diff_par_dist * u_x12;
      double diff_par_y = diff_par_dist * u_y12;
      double diff_par_z = diff_par_dist * u_z12;
      double diff_vert_x = diff_x - diff_par_x;
      double diff_vert_y = diff_y - diff_par_y;
      double diff_vert_z = diff_z - diff_par_z;
      double f12_x = k * (diff_par_x + vert_part * diff_vert_x);
      double f12_y = k * (diff_par_y + vert_part * diff_vert_y);
      double f12_z = k * (diff_par_z + vert_part * diff_vert_z);

      atomicAdd(&f_x[n1], f12_x);
      atomicAdd(&f_y[n1], f12_y);
      atomicAdd(&f_z[n1], f12_z);
      atomicAdd(&f_x[n2], -f12_x);
      atomicAdd(&f_y[n2], -f12_y);
      atomicAdd(&f_z[n2], -f12_z);
      s_sxx -= x12 * f12_x;
      s_sxy -= x12 * f12_y;
      s_sxz -= x12 * f12_z;
      s_syx -= y12 * f12_x;
      s_syy -= y12 * f12_y;
      s_syz -= y12 * f12_z;
      s_szx -= z12 * f12_x;
      s_szy -= z12 * f12_y;
      s_szz -= z12 * f12_z;

    }
    // save virial
    // xx xy xz    0 3 4
    // yx yy yz    6 1 5
    // zx zy zz    7 8 2
    g_virial[n1 + 0 * natoms] += s_sxx;
    g_virial[n1 + 1 * natoms] += s_syy;
    g_virial[n1 + 2 * natoms] += s_szz;
    g_virial[n1 + 3 * natoms] += s_sxy;
    g_virial[n1 + 4 * natoms] += s_sxz;
    g_virial[n1 + 5 * natoms] += s_syz;
    g_virial[n1 + 6 * natoms] += s_syx;
    g_virial[n1 + 7 * natoms] += s_szx;
    g_virial[n1 + 8 * natoms] += s_szy;
  }
}

void read_group_from_target(string target_name, Group& group, Atom& atom, Box& box){
  printf("--------------read target file %s-------------------\n", target_name.data());
  int has_velocity;
  int number_of_types;
  vector<Group> groups;
  GPU_Vector<double> tmp_thermo;
  initialize_position(target_name.data(), has_velocity, number_of_types, box, groups, atom);
  allocate_memory_gpu(groups, atom, tmp_thermo);
  if (groups.size() == 0)
    PRINT_INPUT_ERROR(("there is no group method in " + string(target_name)).data());
  group = groups[0];
  // Group out_group = std::move(groups[0]);
  printf("out group: %d\n", group.cpu_size.front());
}


TargetOpt::TargetOpt()
{
  N1 = 10;
}

void TargetOpt::parse_target_opt(const char** param, int num_param, Force& force)
{
  p_force = &force;
  string target_name = "target.xyz";
  vector<string> target_list;

  for (int n=1; n<num_param; n++){
    if (strcmp(param[n], "k_end") == 0) {
      require_option_values(param, num_param, n, 1, "target_opt");
      if (!is_valid_real(param[n+1], &k_end)) {
        PRINT_INPUT_ERROR("k_end should be a number.");
      }
      n++;
    } else if (strcmp(param[n], "tau") == 0) {
      require_option_values(param, num_param, n, 1, "target_opt");
      if (!is_valid_int(param[n+1], &tau)) {
        PRINT_INPUT_ERROR("Number of steps should be an integer.");
      }
      n++;
    } else if (strcmp(param[n], "rc") == 0) {
      require_option_values(param, num_param, n, 1, "target_opt");
      if (!is_valid_real(param[n+1], &rc)) {
        PRINT_INPUT_ERROR("rc should be a real.");
      }
      n++;
    } else if (strcmp(param[n], "vert_part") == 0) {
      require_option_values(param, num_param, n, 1, "target_opt");
      if (!is_valid_real(param[n+1], &vert_part)) {
        PRINT_INPUT_ERROR("vert_part should be a real.");
      }
      n++;
    } else if (strcmp(param[n], "target") == 0) {
      require_option_values(param, num_param, n, 1, "target_opt");
      target_name = param[n+1];
      n++;
    } else if (strcmp(param[n], "target_list") == 0){
      int i = n + 1;
      for (; i<num_param; i++){
        if (strcmp(param[i], "target_list_end") == 0) break;
        target_list.push_back(string(param[i]));
      }
      if (target_list.empty()) {
        PRINT_INPUT_ERROR("target_list should contain at least one filename.");
      }
      n = i;
    } else if (strcmp(param[n], "max_neighbor") == 0) {
      require_option_values(param, num_param, n, 1, "target_opt");
      if (!is_valid_int(param[n+1], &max_neighbor)) {
        PRINT_INPUT_ERROR("max_neighbor should be an integer.");
      }
      n++;
    } else {
    PRINT_INPUT_ERROR(("no keyword match with: " + string(param[n])).data());
  }
  }
  printf("--------target_opt settings----------\n");
  printf("          rc = %g\n", rc);
  printf("       k_end = %g\n", k_end);
  printf("         tau = %d\n", tau);
  printf("   vert_part = %g\n", vert_part);
  printf("max_neighbor = %d\n", max_neighbor);
  printf("-------------------------------------\n");

  // printf("--------------read target file %s-------------------\n", target_name.data());
  // int has_velocity;
  // int number_of_types;
  // vector<Group> tmp_group;
  // GPU_Vector<double> tmp_thermo;
  // Atom tmp_atom;
  // Box tmp_box;
  // initialize_position(target_name.data(), has_velocity, number_of_types, tmp_box, tmp_group, tmp_atom);
  // allocate_memory_gpu(tmp_group, tmp_atom, tmp_thermo);
  // if (tmp_group.size() == 0)
  //   PRINT_INPUT_ERROR(("there is no group method in " + string(target_name)).data());
  // natoms = tmp_group[0].label.size();
  // Group& group = tmp_group[0];


  // if (group.number <= 1)
  //   PRINT_INPUT_ERROR("there is only one group.");
  // if (group.cpu_size[1] == 0)
  //   PRINT_INPUT_ERROR("there is no atoms in group 1.");
  


  // print_arr(tmp_box.cpu_h, 18, "box");
  // NN_target.resize(natoms); // neighbor number
  // NL_target.resize(natoms * max_neighbor); // neighbor list 
  if (target_list.size() == 0) target_list.push_back(target_name);
  for (auto target_name: target_list){
    // targets.emplace_back(natoms, max_neighbor);
    printf("debug point target_list loop\n");
    targets.emplace_back();
    Target& cur_target = targets.back();
    Atom tmp_atom;
    Box tmp_box;
    read_group_from_target(target_name, cur_target.group, tmp_atom, tmp_box);
    natoms = cur_target.group.label.size();
    GPU_Vector<int> cell_count(natoms);
    GPU_Vector<int> cell_count_sum(natoms);
    GPU_Vector<int> cell_contents(natoms);
    printf("natoms: %d\n", natoms);
    printf("test group: %d\n", cur_target.group.cpu_size.front());
    cur_target.NN.resize(natoms);
    cur_target.NL.resize(natoms * max_neighbor);
    for (int i=0; i<cur_target.group.number-1; i++){
      int n_pick = cur_target.group.cpu_size[i+1];
      cur_target.i_pick_list.emplace_back(n_pick);
      cur_target.dpos_target_list.emplace_back(n_pick * max_neighbor * 3);
      if (n_pick==0) continue;
      auto& i_pick = cur_target.i_pick_list.back();
      auto& dpos_target = cur_target.dpos_target_list.back();
      i_pick.copy_from_device(cur_target.group.contents.data() + cur_target.group.cpu_size_sum[i+1],
        cur_target.group.cpu_size[i+1]);
      // dpos_target.resize(n_pick * max_neighbor * 3);
      find_neighbor(
        0,
        natoms,
        rc,
        tmp_box,
        tmp_atom.type,
        tmp_atom.position_per_atom,
        cell_count,
        cell_count_sum,
        cell_contents,
        cur_target.NN,
        cur_target.NL
      );
      // print_arr(tmp_box.cpu_h, 18, "box");
      // print_gpu(tmp_atom.position_per_atom, "r0");
      // print_gpu(NN_target, "NN0");
      // print_gpu(i_pick, "i_pick");
      // print_gpu(NL_target, "NL_target");
      get_dpos_target<<<(n_pick - 1)/128 + 1, 128>>>(
        natoms,
        tmp_box,
        cur_target.NN.data(),
        cur_target.NL.data(),
        i_pick.data(),
        n_pick,
        tmp_atom.position_per_atom.data(),
        tmp_atom.position_per_atom.data() + natoms,
        tmp_atom.position_per_atom.data() + 2 * natoms,
        dpos_target.data(),
        dpos_target.data() + n_pick * max_neighbor,
        dpos_target.data() + n_pick * max_neighbor * 2
      );
      cudaDeviceSynchronize();
      GPU_CHECK_KERNEL;

    }
    printf("targets size: %zu\n", targets.size());
    printf("target NL: %zu\n", cur_target.NL.size());
  }


  // for (int i=0; i<group.number-1; i++){
  //   int n_pick = group.cpu_size[i+1];
  //   i_pick_list.emplace_back(n_pick);
  //   dpos_target_list.emplace_back(n_pick * max_neighbor * 3);
  //   if (n_pick==0) continue;
  //   auto& i_pick = i_pick_list.back();
  //   auto& dpos_target = dpos_target_list.back();
  //   i_pick.copy_from_device(group.contents.data() + group.cpu_size_sum[i+1], group.cpu_size[i+1]);
  //   // dpos_target.resize(n_pick * max_neighbor * 3);
  //   find_neighbor(
  //     0,
  //     natoms,
  //     rc,
  //     tmp_box,
  //     tmp_atom.type,
  //     tmp_atom.position_per_atom,
  //     cell_count,
  //     cell_count_sum,
  //     cell_contents,
  //     NN_target,
  //     NL_target
  //   );
  //   // print_arr(tmp_box.cpu_h, 18, "box");
  //   // print_gpu(tmp_atom.position_per_atom, "r0");
  //   // print_gpu(NN_target, "NN0");
  //   // print_gpu(i_pick, "i_pick");
  //   // print_gpu(NL_target, "NL_target");
  //   get_dpos_target<<<(n_pick - 1)/128 + 1, 128>>>(
  //     natoms,
  //     0,
  //     natoms,
  //     tmp_box,
  //     NN_target.data(),
  //     NL_target.data(),
  //     i_pick.data(),
  //     n_pick,
  //     tmp_atom.position_per_atom.data(),
  //     tmp_atom.position_per_atom.data() + natoms,
  //     tmp_atom.position_per_atom.data() + 2 * natoms,
  //     dpos_target.data(),
  //     dpos_target.data() + n_pick * max_neighbor,
  //     dpos_target.data() + n_pick * max_neighbor * 2
  //   );
  //   cudaDeviceSynchronize();
  //   GPU_CHECK_KERNEL;
  // }

  force.set_multiple_potentials_mode("sum");

}


void TargetOpt::compute(
  Box& box,
  const GPU_Vector<int>& type,
  const GPU_Vector<double>& position_per_atom,
  GPU_Vector<double>& potential_per_atom,
  GPU_Vector<double>& force_per_atom,
  GPU_Vector<double>& virial_per_atom)
{
  if (type.size() != natoms)
    PRINT_INPUT_ERROR("natoms in model.xyz and target does not match");

  step++;

  double k = step < tau ? k_end * step / tau : k_end;
  // printf("k = %f\n", k);
  
  for (auto& target:targets){
    
    if (is_small_box){

    }
    else {
      for (size_t i=0; i<target.i_pick_list.size(); i++){
        auto& i_pick = target.i_pick_list[i];
        auto& dpos_target = target.dpos_target_list[i];
        int n_pick = i_pick.size();
        if (n_pick==0) continue;
        calc_spring_force<<<(n_pick-1)/128+1, 128>>>(
          natoms,
          box,
          target.NN.data(),
          target.NL.data(),
          i_pick.data(),
          n_pick,
          k / pow(2.0, i),
          vert_part,
          position_per_atom.data(),
          position_per_atom.data() + natoms,
          position_per_atom.data() + natoms * 2,
          dpos_target.data(),
          dpos_target.data() + n_pick * max_neighbor,
          dpos_target.data() + n_pick * max_neighbor * 2,
          force_per_atom.data(),
          force_per_atom.data() + natoms,
          force_per_atom.data() + natoms * 2,
          virial_per_atom.data()
        );
      }
    }
  }
}
