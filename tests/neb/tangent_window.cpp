#include "neb/tangent_window.h"
#include "neb/ode12r.h"
#include <cassert>
#include <cstdio>
#include <limits>
int main(int argc, char** argv) {
  assert(argc==2);
  NEBTangentWindow::Gate g(3,.015,.03);
  assert(g.inactive());
  assert(g.update({.04,.02,.01}));
  assert(g.active(0) && !g.active(1));
  assert(!g.update({.03,.03,.015})); // thresholds retain the previous state
  const int retained=g.ids[0];g.insert(1);
  assert(g.ids[0]==retained && g.active(0) && !g.active(1));
  g.erase(2);assert(g.force.size()==3 && g.generation==2);
  assert(g.update({.014,.05,.02}));
  g.save(argv[1],672);
  NEBTangentWindow::Gate h(3,.015,.03);h.load(argv[1],672);
  assert(h.ids==g.ids && h.force==g.force && h.switches==g.switches && h.next_id==g.next_id);
  bool failed=false;try {h.load(argv[1],2);}catch(...){failed=true;}assert(failed);
  failed=false;try {h.update({.01,std::numeric_limits<double>::quiet_NaN(),.01});}catch(...){failed=true;}assert(failed);
  std::remove(argv[1]);
  auto accepted=NEBODE12r::decide(.02,1.,.9,.001,.1,.2,.01,1e-10,true,.1);
  auto rejected=NEBODE12r::decide(.02,.001,.004,.000001,.1,.2,.01,1e-10,true,.1);
  assert(accepted.accept && !rejected.accept && rejected.next_h<.02);
}
