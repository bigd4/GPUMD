#include "atoms.cuh"
#include <unistd.h> // For UNIX/Linux systems

// GPU_Vector<double>& Atoms::get_forces() 
// {
//     force.compute(
//         box, positions, type, group, potentials, forces, virials);

//     return forces; 
// }

Atoms::Atoms(const Atoms& atoms0) {
    box = atoms0.box;
    positions = atoms0.positions;
    type = atoms0.type;
    group = atoms0.group;
    potential_per_atom = atoms0.potential_per_atom;
    forces = atoms0.forces;
    virials = atoms0.virials;
}

Atoms::Atoms(
  Box& _box,
  GPU_Vector<double>& _positions,
  GPU_Vector<int>& _type,
  vector<Group>& _group,
  GPU_Vector<double>& _potentials,
  GPU_Vector<double>& _forces,
  GPU_Vector<double>& _virials)
{
    box = _box;
    positions = _positions;
    type = _type;
    group = _group;
    potential_per_atom = _potentials;
    forces = _forces;
    virials = _virials;
}

Atoms::Atoms(Atom& atom) {
    natoms = atom.number_of_atoms;
    positions = atom.position_per_atom;
    type = atom.type;
    group = group;
    potential_per_atom = atom.potential_per_atom;
    forces = atom.force_per_atom;
    virials = atom.virial_per_atom;
}

void Atoms::compute() { p_force->compute(box, positions, type, group, potential_per_atom, forces, virials); }

GPU_Vector<double>& Atoms::get_virial() {
    return virials;
}

void Atoms::set_positions() {}


Atoms xyz2atoms()
{
  int has_velocity;
  int number_of_types;
  Box _box;
  vector<Group> _group;
  Atom atom;
  Atoms atoms;
  GPU_Vector<double> thermo;


  initialize_position("model.xyz", has_velocity, number_of_types, _box, _group, atom);

  printf("finish initialize position\n");

  allocate_memory_gpu(_group, atom, thermo);

  printf("finish allocate memory\n");

  atoms.natoms = atom.number_of_atoms;
  atoms.positions = atom.position_per_atom;
  atoms.type = atom.type;
  atoms.group = _group;
  atoms.potential_per_atom = atom.potential_per_atom;
  atoms.forces = atom.force_per_atom;
  atoms.virials = atom.virial_per_atom;
  printf("natoms: %d\n", atoms.natoms);
  sleep(2);
  return atoms;
}
