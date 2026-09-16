#include "neb/energy_window.h"
#include <cassert>
#include <limits>
int main() {
  neb_energy::Window w; double range;
  assert(!w.update(0, 1., 3, .1, range));
  assert(!w.update(1, 2., 3, .1, range));
  assert(!w.update(2, 1., 3, .1, range)); // matching endpoints do not hide oscillation
  assert(!w.update(3, 1., 3, .1, range));
  assert(w.update(4, 1., 3, .1, range));
  w.clear(); // topology/local-stage reset
  assert(!w.update(5, 1., 3, .1, range));
  assert(!w.update(7, 1., 3, .1, range)); // missing step resets history
  assert(!w.update(8, 1., 3, .1, range));
  assert(w.update(9, 1., 3, 0., range));
  assert(!w.update(10, std::numeric_limits<double>::quiet_NaN(), 3, .1, range));
  assert(!w.update(11, 1., 3, .1, range));
  w.clear();
  assert(!w.update(0, 1., 2, .125, range));
  assert(w.update(1, 1.125, 2, .125, range)); // inclusive threshold
  assert(!w.update(2, 1., 0, .125, range));
}
