#pragma once
#include <algorithm>
#include <cmath>
#include <limits>

namespace NEBODE12r {
struct Decision { bool accept; double next_h, h_error, h_ls; };
// ASE ode12r, extrapolate=3. Error and residual use the caller's metric.
inline Decision decide(double h, double old_r, double new_r, double error,
                       double fy, double yy, double rtol, double hmin, bool residual_extrapolation = true, double secant_guard = 0.0) {
  const bool finite = std::isfinite(new_r) && std::isfinite(error) &&
                      std::isfinite(fy) && std::isfinite(yy);
  const bool accept = finite && ((new_r <= old_r * (1.0 - 0.01*h)) ||
                                (new_r <= 2.0*old_r && error <= rtol));
  double hls = finite ? h * fy / (yy + 1e-10) : hmin;
  if (!std::isfinite(hls) || hls < hmin) hls = std::numeric_limits<double>::infinity();
  if (!residual_extrapolation) hls = std::numeric_limits<double>::infinity();
  const double herr = !finite ? 0.1*h : (error > 0.0 ? 0.5*h*std::sqrt(rtol/error)
                                      : std::numeric_limits<double>::infinity());
  // Bound the residual predictor relative to the error-controlled step, not
  // an absolute minimum displacement. Trial acceptance remains unchanged.
  if (secant_guard > 0.0 && residual_extrapolation)
    hls = std::max(hls, secant_guard * herr);
  const double next = accept ? std::max(0.25*h, std::min(4.0*h, std::min(herr,hls)))
                             : std::max(0.1*h, std::min(0.25*h, std::min(herr,hls)));
  return {accept, next, herr, hls};
}
struct StagnationMonitor {
  int small_steps=0, cooldown=0, triggers=0;
  double start_residual=0.0, best_residual=0.0;
  void observe(bool accepted, double move, double old_r, double new_r, double tol) {
    if (cooldown>0) --cooldown;
    if (!accepted) return;
    if (move>=1e-7 || new_r<=tol) { small_steps=0; return; }
    if (small_steps==0) { start_residual=old_r; best_residual=old_r; }
    best_residual=std::min(best_residual,new_r);
    if (++small_steps>=20) {
      if (best_residual>=0.99*start_residual) { cooldown=50; ++triggers; }
      small_steps=0;
    }
  }
};
}
