#include "io/output_format.h"
#include <stdexcept>
int main() {
  trajectory::Config c;
  c.parse({"output_format","xyz"});
  if (trajectory::filename("velocity.out", c)!="velocity.out") return 1;
  try {c.parse({"output_format","bxyz"});} catch(const std::runtime_error&) {return 0;}
  return 1;
}
