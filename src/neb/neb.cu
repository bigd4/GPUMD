#include "neb.cuh"
#include <unistd.h> // For UNIX/Linux systems

NEB::NEB(){
  // variable_cell = false;
  double pressure[] = {1.0, 1.0, 1.0};
  Atoms *p_atoms = new Atoms("neb_fs.xyz");
  if (!variable_cell){
    images.push_back(p_atoms);
    printf("neb() images[0] natoms %d\n", images[0]->natoms);
    // Atoms *p_atoms3 = new Atoms("neb_fs.xyz");
    // images.push_back(p_atoms3);
  }
  else{
    VCWrapper *p_atoms2 = new VCWrapper(*p_atoms, pressure, 3);
    images.push_back(static_cast<VCWrapper*> (p_atoms2));
    printf("neb() images[0] natoms %d\n", dynamic_cast<VCWrapper*>(images[0])->natoms);
  }
  printf("neb default construtor finishes.\n");
}


void NEB::parse_neb(const char** param, int num_param, Force& force)
{

  p_force = &force;
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

  switch (minimizer_type) {
    case 1:
      printf("\nStart to do neb calculation.\n");
      printf("    using the fast inertial relaxation engine (FIRE) method.\n");
      printf("    with fixed box.\n");
      printf("    with a force tolerance of %g eV/A.\n", force_tolerance);
      printf("    for maximally %d steps.\n", max_steps);

      if (!variable_cell){
        printf("normal neb\n");
        norm_neb();
      }
      else{
        printf("variable cell neb\n");
        vcneb();
      }
      dump_position.postprocess();
      break;
    default:
      PRINT_INPUT_ERROR("Invalid minimizer.");
      break;
  }
}

void NEB::compute() {
  for (int i=0; i < images.size(); i++){
    Atoms& image = *images[i];
    dump_position.process(1, image.box, image.group, image.cpu_atom_symbol, image.cpu_type,
        image.get_positions(), image.cpu_positions);
  }
}

GPU_Vector<double>& NEB::get_positions()
{
  return positions;
}

GPU_Vector<double>& NEB::get_forces()
{
  // TODO: insert return statement here
}

// void test(BaseAtoms& atoms){
//   printf("-------go in test-----------\n");
//   atoms.get_positions();
//   printf("-------go out of test-----------\n");
// }

void NEB::norm_neb() {
  unique_ptr<Minimizer> minimizer;
  Atoms& image = *images[0];

        

  const char* para[] = {"a","1"};
  image.set_calc(*p_force);
  minimizer.reset(new Minimizer_FIRE_JQH(image.natoms, max_steps, force_tolerance));
  printf("k = %f\n", k); 
  dump_position.parse(para, 2, image.group);
  dump_position.preprocess();
  // printf("image addr: %p\n", &image);
  minimizer->compute(image);
}

void NEB::vcneb() {
  
  unique_ptr<Minimizer> minimizer;
  // GPU_Vector<double> *pos;
  VCWrapper& image = *static_cast<VCWrapper *>(images[0]);
  // pos = &image.get_positions();
  // (*dynamic_cast<BaseAtoms *>(images[0])).get_positions();
  const char* para[] = {"a","1"};
  // printf("parse_neb natoms %d\n", images[0]->natoms);
  printf("parse_neb image natoms %d\n", image.natoms);
  // image.compute();
  image.set_calc(*p_force);
  minimizer.reset(new Minimizer_FIRE_JQH(image.natoms, max_steps, force_tolerance));
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
