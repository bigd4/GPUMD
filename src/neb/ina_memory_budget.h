#pragma once

#include <cmath>
#include <cstddef>

namespace ina_memory {

// Conservative extra allocation budget, not a measurement of process usage.
// Do not credit live NEB/FIRE buffers: their lifetime overlaps INA insertion.
// Include the full future workspace on every check so multiple insertions in
// one INA pass cannot spend the memory needed for the deferred rebuild.
inline long double extra_bytes(
  std::size_t image_bytes, std::size_t atoms_per_image, std::size_t future_images)
{
  const long double movable = future_images > 2 ? future_images - 2 : 0;
  const long double dofs = 3.0L * atoms_per_image;
  // Global NEB arrays, either FIRE variant, DYNEB and transient vector copies.
  const long double workspace =
    (12.0L * dofs * movable + 32.0L * future_images) * sizeof(double);
  return 1.25L * (static_cast<long double>(image_bytes) + workspace);
}

inline bool fits(std::size_t free_bytes, long double extra, double reserve_mib)
{
  if (!std::isfinite(extra) || extra < 0 ||
      !std::isfinite(reserve_mib) || reserve_mib < 0) return false;
  const long double reserve = static_cast<long double>(reserve_mib) * 1024 * 1024;
  return static_cast<long double>(free_bytes) >= extra + reserve;
}

} // namespace ina_memory
