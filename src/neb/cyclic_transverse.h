#pragma once
#include <cmath>

namespace NEBCyclicTransverse {
struct Settings {
  int parallel_steps = 5;
  int perpendicular_steps = 50;
  int settling_steps = 220;
  int required_cycles = 10;
};

// Counts force evaluations, not displacements. Topology changes invalidate
// all acceptance evidence even if an insertion and deletion preserve size.
struct State {
  bool parallel = true;
  int stable_steps = 0;
  int cycles = 0;

  bool settled(const Settings& settings) const { return stable_steps >= settings.settling_steps; }

  bool finish(const Settings& settings, int evaluations, bool changed,
              double live, double verified, double tolerance)
  {
    if (changed) {
      stable_steps = 0;
      cycles = 0;
    } else {
      stable_steps += evaluations;
    }
    const bool accepted = !parallel && !changed && settled(settings) &&
      std::isfinite(live) && std::isfinite(verified) && live <= tolerance && verified <= tolerance;
    if (accepted) ++cycles;
    parallel = parallel ? false : (accepted || !settled(settings));
    return accepted;
  }
};
}
