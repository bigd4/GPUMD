#include "neb.cuh"
#include <unistd.h> // For UNIX/Linux systems

NEB::NEB(){
  Atoms *p_atoms = new Atoms();
  *p_atoms = xyz2atoms();
  printf("build neb atoms success\n");
  sleep(2);
  images.push_back(p_atoms);
  printf("neb default construtor finishes.\n");
  sleep(2);
}

void NEB::parse_neb(
  const char** param,
  int num_param)
{

  int minimizer_type = 0;
  int number_of_steps = 0;
  bool vc = false;
  double pressure = 0.0;
  double force_tolerance = 0.0;
  unique_ptr<Minimizer> minimizer;
  const int number_of_atoms = images[0]->natoms;




  // images.push_back(Atoms())


  if (strcmp(param[1], "sd") == 0) {
    minimizer_type = 0;

    if (num_param != 4) {
      PRINT_INPUT_ERROR("minimize sd should have 2 parameters.");
    }

    if (!is_valid_real(param[2], &force_tolerance)) {
      PRINT_INPUT_ERROR("Force tolerance should be a number.");
    }

    if (!is_valid_int(param[3], &number_of_steps)) {
      PRINT_INPUT_ERROR("Number of steps should be an integer.");
    }
    if (number_of_steps <= 0) {
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

    if (!is_valid_int(param[3], &number_of_steps)) {
      PRINT_INPUT_ERROR("Number of steps should be an integer.");
    }

    if (!is_valid_real(param[4], &k)) {
      PRINT_INPUT_ERROR("Number of steps should be an real.");
    }
    if (number_of_steps <= 0) {
      PRINT_INPUT_ERROR("Number of steps should > 0.");
    }
  }

  switch (minimizer_type) {
    // case 0:
      // printf("\nStart to do neb calculation.\n");
    //   printf("    using the steepest descent method.\n");
    //   printf("    with fixed box.\n");
    //   printf("    with a force tolerance of %g eV/A.\n", force_tolerance);
    //   printf("    for maximally %d steps.\n", number_of_steps);

    //   minimizer.reset(new Minimizer_SD(number_of_atoms, number_of_steps, force_tolerance));

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
      printf("    for maximally %d steps.\n", number_of_steps);

      minimizer.reset(new Minimizer_FIRE(number_of_atoms, number_of_steps, force_tolerance));
      printf("k = %f\n", k);


    //   minimizer->compute(
    //     force,
    //     box,
    //     position_per_atom,
    //     type,
    //     group,
    //     potential_per_atom,
    //     force_per_atom,
    //     virial_per_atom);

      break;
    default:
      PRINT_INPUT_ERROR("Invalid minimizer.");
      break;
  }
}

