#pragma once
#include <algorithm>
#include <cmath>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace NEBTangentWindow {
// Only accepted force evaluations update this hysteresis. Geometry never opens it.
struct Gate {
  double off, on;
  int next_id = 1, generation = 0;
  std::vector<int> ids, force, geometry, switches; // geometry is always zero (log compatibility).
  Gate(int n, double low, double high)
    : off(low), on(high), force(n, 0), geometry(n, 0), switches(n, 0)
  {
    if (!(std::isfinite(low) && std::isfinite(high) && low > 0 && high > low))
      throw std::invalid_argument("Require finite 0 < window_off < window_on.");
    for (int i = 0; i < n; ++i) ids.push_back(next_id++);
  }
  void insert(int i)
  {
    force.insert(force.begin() + i, 0);
    geometry.insert(geometry.begin() + i, 0);
    switches.insert(switches.begin() + i, 0);
    ids.insert(ids.begin() + i, next_id++);
    ++generation;
  }
  void erase(int i)
  {
    force.erase(force.begin() + i);
    geometry.erase(geometry.begin() + i);
    switches.erase(switches.begin() + i);
    ids.erase(ids.begin() + i);
    ++generation;
  }
  bool active(int i) const { return force[i] != 0; }
  std::vector<int> mask() const { return force; }
  bool inactive() const { return std::none_of(force.begin(), force.end(), [](int v) { return v; }); }
  bool update(const std::vector<double>& values)
  {
    if (values.size() != force.size()) throw std::invalid_argument("Window force/image count mismatch.");
    bool changed = false;
    for (size_t i = 0; i < force.size(); ++i) {
      if (!std::isfinite(values[i]) || values[i] < 0)
        throw std::invalid_argument("Nonfinite or negative window residual.");
      const int old = force[i];
      if (values[i] > on) force[i] = 1;
      else if (values[i] < off) force[i] = 0;
      if (old != force[i]) { changed = true; ++switches[i]; }
    }
    return changed;
  }
  // A restart retains image identities and latches, but not convergence streaks.
  void save(const std::string& path, int atoms) const
  {
    std::ofstream out(path);
    out.precision(17);
    out << "NEB_WINDOW_V1 " << force.size() << ' ' << atoms << ' ' << off << ' ' << on << ' ' << next_id << '\n';
    for (size_t i = 0; i < force.size(); ++i)
      out << ids[i] << ' ' << force[i] << ' ' << switches[i] << '\n';
    if (!out) throw std::runtime_error("Cannot write window checkpoint.");
  }
  void load(const std::string& path, int atoms)
  {
    std::ifstream in(path);
    std::string tag, extra;
    int count, atom_count, next;
    double low, high;
    if (!(in >> tag >> count >> atom_count >> low >> high >> next) || tag != "NEB_WINDOW_V1" ||
        count != int(force.size()) || atom_count != atoms || low != off || high != on || next <= 0)
      throw std::runtime_error("Window restart header, thresholds, or path size mismatch.");
    std::set<int> seen;
    auto new_ids = ids, new_force = force, new_switches = switches;
    for (int i = 0; i < count; ++i) {
      if (!(in >> new_ids[i] >> new_force[i] >> new_switches[i]) || new_ids[i] <= 0 ||
          new_ids[i] >= next || !seen.insert(new_ids[i]).second ||
          (new_force[i] != 0 && new_force[i] != 1) || new_switches[i] < 0)
        throw std::runtime_error("Invalid window restart image state.");
    }
    if (in >> extra) throw std::runtime_error("Trailing window restart data.");
    ids.swap(new_ids); force.swap(new_force); switches.swap(new_switches); next_id = next;
  }
};
}
