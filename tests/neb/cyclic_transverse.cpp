#include "neb/cyclic_transverse.h"
#include <cassert>
#include <limits>

int main()
{
  using namespace NEBCyclicTransverse;
  Settings options;
  State state;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  // Spring-only zero force must never be accepted.
  assert(!state.finish(options, 5, false, 0.0, 0.0, .001));
  assert(!state.parallel && state.cycles == 0);
  assert(!state.finish(options, 50, false, .0001, .0001, .001));
  assert(state.parallel && state.stable_steps == 55);
  // Insertion plus deletion can preserve size but must invalidate evidence.
  state.cycles = 3;
  state.stable_steps = 300;
  state.parallel = false;
  assert(!state.finish(options, 50, true, .0001, .0001, .001));
  assert(state.cycles == 0 && state.stable_steps == 0 && state.parallel);
  state.parallel = false;
  state.stable_steps = 220;
  assert(!state.finish(options, 1, false, .0009, .0011, .001));
  assert(!state.parallel && state.cycles == 0);
  assert(!state.finish(options, 1, false, .0009, nan, .001));
  for (int i = 1; i <= 10; ++i) {
    assert(state.finish(options, 1, false, .001, .001, .001));
    assert(state.cycles == i && state.parallel);
    if (i != 10) {
      assert(!state.finish(options, 5, false, 0.0, nan, .001));
      assert(!state.parallel);
    }
  }
}
