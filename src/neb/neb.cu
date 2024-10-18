#include "neb.cuh"
#include <unistd.h> // For UNIX/Linux systems

NEB::NEB(){
  // Atoms *p_atoms = new Atoms(xyz2atoms());
  // Atoms *p_atoms = new Atoms();
  // Atoms *p_atoms;
  // Atoms &a=xyz2atoms();
  // p_atoms = &a;
  // Atoms *p_atoms = new Atoms();
  // xyz2atoms(*p_atoms);
  // Atoms *p_atoms = new Atoms(xyz2atoms());
  Atoms *p_atoms = new Atoms("model.xyz");
  printf("move constructor addr: %p\n", p_atoms);
  printf("build neb atoms success\n");
  images.push_back(p_atoms);
  printf("neb default construtor finishes.\n");
  // sleep(2);
}

void NEB::parse_neb(const char** param, int num_param, Force& force)
{
  int minimizer_type = 0;
  int max_steps = 0;
  bool vc = false;
  double pressure = 0.0;
  double force_tolerance = 0.0;
  unique_ptr<Minimizer> minimizer;
  const int number_of_atoms = images[0]->natoms;

  const char* para[] = {"a","1"};
  Atoms& image = *images[0];
  image.set_calc(force);


  if (strcmp(param[1], "sd") == 0) {
    minimizer_type = 0;

    if (num_param != 4) {
      PRINT_INPUT_ERROR("minimize sd should have 2 parameters.");
    }

    if (!is_valid_real(param[2], &force_tolerance)) {
      PRINT_INPUT_ERROR("Force tolerance should be a number.");
    }

    if (!is_valid_int(param[3], &max_steps)) {
      PRINT_INPUT_ERROR("Number of steps should be an integer.");
    }
    if (max_steps <= 0) {
      PRINT_INPUT_ERROR("Number of steps should > 0.");
    }
  } else if (strcmp(param[1], "fire") == 0) {
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

  switch (minimizer_type) {
    // case 0:
      // printf("\nStart to do neb calculation.\n");
    //   printf("    using the steepest descent method.\n");
    //   printf("    with fixed box.\n");
    //   printf("    with a force tolerance of %g eV/A.\n", force_tolerance);
    //   printf("    for maximally %d steps.\n", max_steps);

    //   minimizer.reset(new Minimizer_SD(number_of_atoms, max_steps, force_tolerance));

    //   minimizer->compute(
    //     force,
    //     box,
    //     position_per_atom,
    //     type,
    //     group,
    //     potential_per_atom,
    //     force_per_atom,
    //     virial_per_atom);

    //   break;
    case 1:
      if (vc){
        printf("variable cell is enabled");
        }
      printf("\nStart to do neb calculation.\n");
      printf("    using the fast inertial relaxation engine (FIRE) method.\n");
      printf("    with fixed box.\n");
      printf("    with a force tolerance of %g eV/A.\n", force_tolerance);
      printf("    for maximally %d steps.\n", max_steps);

      minimizer.reset(new Minimizer_FIRE_JQH(number_of_atoms, max_steps, force_tolerance));
      printf("k = %f\n", k);
      dump_position.parse(para, 2, image.group);
      dump_position.preprocess();

      // images[0]->compute();
      image.compute();
      
      // sleep(2);
      double pos[5];
      // image.positions.copy_to_host(pos, 5);
      // printf("print pos %f\n", pos[3]);
      // for (int step=0; step < max_steps; step++){
      //   one_neb_step();
      // }

      dump_position.postprocess();

      break;
    default:
      PRINT_INPUT_ERROR("Invalid minimizer.");
      break;
  }
}

// void NEB::one_neb_step(){
//   for (int i; i < images.size(); i++){
//     Atoms& image = *images[i];
//     dump_position.process(1, image.box, image.group, image.cpu_atom_symbol, image.cpu_type,
//                           image.get_positions(), image.cpu_positions);

// }

// }


GPU_Vector<double>& NEB::get_positions()
{
  for (int i=0; i < images.size(); i++){
    Atoms& image = *images[i];
    dump_position.process(1, image.box, image.group, image.cpu_atom_symbol, image.cpu_type,
                        image.get_positions(), image.cpu_positions);
  }

  return positions;
}
