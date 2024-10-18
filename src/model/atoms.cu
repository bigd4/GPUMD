#include "atoms.cuh"
#include <unistd.h> // For UNIX/Linux systems

// GPU_Vector<double>& Atoms::get_forces() 
// {
//     force.compute(
//         box, positions, type, group, potentials, forces, virials);

//     return forces; 
// }

Atoms::Atoms() {
    printf("atoms default construtor for %p\n", this);
}

// Atoms::Atoms(const Atoms& atoms0)
// {
//   printf("start atoms copy construct");
//   natoms = atoms0.natoms;
//   energy = atoms0.energy;
//   p_force = atoms0.p_force;
//   cpu_atom_symbol = atoms0.cpu_atom_symbol;
//   box = atoms0.box;
//   positions = atoms0.positions;
//   type = atoms0.type;
//   group = atoms0.group;
//   potential_per_atom = atoms0.potential_per_atom;
//   forces = atoms0.forces;
//   virials = atoms0.virials;
//   printf("finish atoms copy construct");
// }

Atoms::~Atoms() {
    printf("atoms destructor for %p\n", this);
}

// Atoms::Atoms(
//   Box& _box,
//   GPU_Vector<double>& _positions,
//   GPU_Vector<int>& _type,
//   vector<Group>& _group,
//   GPU_Vector<double>& _potentials,
//   GPU_Vector<double>& _forces,
//   GPU_Vector<double>& _virials)
// {
//     box = _box;
//     positions = _positions;
//     type = _type;
//     group = _group;
//     potential_per_atom = _potentials;
//     forces = _forces;
//     virials = _virials;
// }

// Atoms::Atoms(Atom& atom) {
//     natoms = atom.number_of_atoms;
//     positions = atom.position_per_atom;
//     type = atom.type;
//     group = group;
//     potential_per_atom = atom.potential_per_atom;
//     forces = atom.force_per_atom;
//     virials = atom.virial_per_atom;
// }

Atoms::Atoms(const char* filename)
{
  printf("file %s to atoms.\n", filename);
  int has_velocity;
  int number_of_types;
  Atoms atoms;
  // Box _box;
  // vector<Group> _group;
  GPU_Vector<double> thermo;
  Atom atom;

  initialize_position(filename, has_velocity, number_of_types, box, group, atom);
  allocate_memory_gpu(group, atom, thermo);

  natoms = atom.number_of_atoms;
  cpu_atom_symbol = move(atom.cpu_atom_symbol);
  cpu_positions = move(atom.cpu_position_per_atom);
  type = move(atom.type);
  positions = move(atom.position_per_atom);
  potential_per_atom = move(atom.potential_per_atom);
  forces = move(atom.force_per_atom);
  virials = move(atom.virial_per_atom);
}

void Atoms::compute()
{
  if (changed_after_last_compute) {
    p_force->compute(box, positions, type, group, potential_per_atom, forces, virials);
    changed_after_last_compute = false;
  }
}

GPU_Vector<double>& Atoms::get_virials()
{
  if (changed_after_last_compute) compute();
  return virials;
}

GPU_Vector<double>& Atoms::get_virial() { return virials; }

void Atoms::set_positions() {}

GPU_Vector<double>& Atoms::get_potential_per_atom()
{
  if (changed_after_last_compute) compute();
  return potential_per_atom;
}

GPU_Vector<double>& Atoms::get_forces()
{
  if (changed_after_last_compute) compute();
  return forces;
}

void Atoms::set_calc(Force& force)
{
  printf("set calc\n");
  p_force = &force;
}

void xyz2atoms(Atoms& atoms)
{
    printf("xyz2atoms start ---------------\n");
  int has_velocity;
  int number_of_types;
  Box _box;
  vector<Group> _group;
  GPU_Vector<double> thermo;
  Atom atom;

  initialize_position("model.xyz", has_velocity, number_of_types, _box, _group, atom);
  allocate_memory_gpu(_group, atom, thermo);

  // printf("atom pot %p\n", &atoms._atom.potential_per_atom);
  // GPU_Vector<double>& mypos = GPU_Vector_copy(atoms._atom.position_per_atom);
  // printf("test gpu vec copy\n");

  atoms.natoms = atom.number_of_atoms;
  // atoms.positions = atoms._atom.position_per_atom;
  // atoms.type = atoms._atom.type;
  // atoms.group = _group;
  // atoms.potential_per_atom = atoms._atom.potential_per_atom;
  // atoms.forces = atoms._atom.force_per_atom;
  // atoms.virials = atoms._atom.virial_per_atom;
  printf("natoms: %d\n", atoms.natoms);
}

// Atoms xyz2atoms()
// {
//   printf("xyz2atoms start ---------------\n");
//   int has_velocity;
//   int number_of_types;
//   Atoms atoms;
//   // Box _box;
//   // vector<Group> _group;
//   GPU_Vector<double> thermo;
//   Atom atom;

//   initialize_position("model.xyz", has_velocity, number_of_types, atoms.box, atoms.group, atom);
//   allocate_memory_gpu(atoms.group, atom, thermo);


//   atoms.natoms = atom.number_of_atoms;
//   atoms.cpu_atom_symbol = move(atom.cpu_atom_symbol);
//   atoms.cpu_positions = move(atom.cpu_position_per_atom);
//   atoms.type = move(atom.type);
//   atoms.positions = move(atom.position_per_atom);
//   atoms.potential_per_atom = move(atom.potential_per_atom);
//   atoms.forces = move(atom.force_per_atom);
//   atoms.virials = move(atom.virial_per_atom);
//   printf("natoms: %d\n", atoms.natoms);
//   return atoms;
// }
