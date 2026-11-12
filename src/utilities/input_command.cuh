#pragma once
#include <istream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace GPUMDInput {
// Read one logical run.in command. Keep physical-line readers for XYZ/potentials unchanged.
inline std::vector<std::string> read_command(std::istream& input)
{
  std::vector<std::string> command;
  std::string line;
  bool continued = false;
  while (std::getline(input, line)) {
    std::istringstream words(line);
    std::vector<std::string> tokens;
    std::string token;
    while (words >> token) {
      // Match the existing run.in comment rule: # starts a comment token.
      if (token[0] == '#') break;
      tokens.push_back(token);
    }
    if (tokens.empty()) {
      if (continued) continue;
      return command;
    }
    const bool more = tokens.back() == "\\" || tokens.back() == "&";
    if (more) tokens.pop_back();
    command.insert(command.end(), tokens.begin(), tokens.end());
    if (!more) return command;
    continued = true;
  }
  if (continued)
    throw std::runtime_error("Unterminated input continuation: expected command content after the trailing \\ or & before end of file.");
  return command;
}
}
