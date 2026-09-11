#include "ina_memory_budget.h"
#include <cassert>
#include <limits>

int main()
{
  using namespace ina_memory;
  const std::size_t mib = 1024 * 1024;
  assert(fits(513 * mib, mib, 512));
  assert(!fits(513 * mib - 1, mib, 512));
  assert(fits(mib, mib, 0));
  assert(!fits(mib, -1, 0));
  assert(!fits(mib, std::numeric_limits<long double>::infinity(), 0));
  assert(!fits(mib, 0, std::numeric_limits<double>::quiet_NaN()));
  assert(!fits(mib, 0, -1));
  const auto a = extra_bytes(1000000, 6403, 10);
  const auto b = extra_bytes(1000000, 6403, 11);
  assert(b > a);
  assert(extra_bytes(2000000, 6403, 10) > a);
  assert(extra_bytes(1000000, 12803, 10) > a);
  // No narrowing overflow when large image counts are used in the estimate.
  assert(extra_bytes(1000000, 6403, std::numeric_limits<std::size_t>::max()) > b);
  return 0;
}
