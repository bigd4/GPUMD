#include "utilities/input_command.cuh"
#include <cassert>
#include <sstream>
#include <vector>
#include <string>
using Tokens = std::vector<std::string>;
static std::vector<Tokens> parse(const std::string& text)
{
  std::istringstream in(text);
  std::vector<Tokens> result;
  while (in.peek() != EOF) {
    auto tokens=GPUMDInput::read_command(in);
    if (!tokens.empty()) result.push_back(tokens);
  }
  return result;
}
int main()
{
  const auto expected=parse("potential nep.txt\nneb_run fire .005 3000 dt_max 1 max_move .1\n");
  assert(parse("potential \\\n nep.txt\nneb_run fire .005 3000 \\\n dt_max 1 \\\n max_move .1\n")==expected);
  assert(parse("potential &\n nep.txt\nneb_run fire .005 3000 &\n dt_max 1 &\n max_move .1")==expected);
  assert(parse("potential \\ # model\n # ignored &\n\n nep.txt\nneb_run fire .005 3000 & \t# options\r\n\r\n dt_max 1 \\\r\n max_move .1 # stop\r\n")==expected);
  assert(parse("# comment \\\n\npotential nep.txt # &\n")==std::vector<Tokens>({{"potential","nep.txt"}}));
  assert(parse("potential path&\npotential file#part.txt\n")==std::vector<Tokens>({{"potential","path&"},{"potential","file#part.txt"}}));
  assert(parse("potential path\\\n")==std::vector<Tokens>({{"potential","path\\"}}));
  assert(parse("a & b\n")==std::vector<Tokens>({{"a","&","b"}}));
  for (const auto& text : {"potential \\","potential &\n","potential \\ # comment\n # only a comment\n\n"}) {
    bool failed=false;
    try { parse(text); } catch (const std::runtime_error&) { failed=true; }
    assert(failed);
  }
  assert(parse("").empty());
  assert(parse("\n# comment\n").empty());
}
