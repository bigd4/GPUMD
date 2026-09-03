#pragma once

#include <cctype>
#include <string>

namespace gas_config {

inline void trim(std::string& value)
{
  size_t first = 0;
  while (first < value.size() &&
         std::isspace(static_cast<unsigned char>(value[first]))) {
    ++first;
  }
  size_t last = value.size();
  while (last > first &&
         std::isspace(static_cast<unsigned char>(value[last - 1]))) {
    --last;
  }
  value = value.substr(first, last - first);
}

inline void unquote(std::string& value)
{
  if (value.size() >= 2 &&
      ((value.front() == '"' && value.back() == '"') ||
       (value.front() == 39 && value.back() == 39))) {
    value = value.substr(1, value.size() - 2);
  }
}

} // namespace gas_config
