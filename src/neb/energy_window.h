#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>

namespace neb_energy {
// One sample per consecutive regular NEB iteration; local/changed paths reset it.
class Window {
  std::deque<double> samples_;
  int last_step_ = -1;
public:
  void clear() { samples_.clear(); last_step_ = -1; }
  bool update(int step, double energy, int length, double tolerance, double& spread) {
    spread = 0.0;
    if (length < 2 || !std::isfinite(energy)) { clear(); return false; }
    if (last_step_ != step - 1) clear();
    last_step_ = step;
    samples_.push_back(energy);
    if (samples_.size() > static_cast<std::size_t>(length)) samples_.pop_front();
    if (samples_.size() < static_cast<std::size_t>(length)) return false;
    auto bounds = std::minmax_element(samples_.begin(), samples_.end());
    spread = *bounds.second - *bounds.first;
    return std::isfinite(spread) && spread <= tolerance;
  }
};
}
