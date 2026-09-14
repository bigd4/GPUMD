#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace binary_xyz {

class Error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Flat atom-major arrays: atom 0's components, then atom 1's components, ...
using Values = std::variant<std::vector<double>, std::vector<std::int64_t>,
                            std::vector<std::uint8_t>, std::vector<std::string>>;
struct Property {
    std::string name;
    std::size_t components = 1;
    Values values;
};

struct Frame {
    std::vector<std::uint8_t> numbers;   // 0=X, 1=H, ... 118=Og
    std::vector<double> positions;      // 3 * numbers.size()
    std::array<double, 9> cell{};        // Row-major ASE cell convention
    std::array<bool, 3> pbc{false, false, false};
    std::string info_extxyz;            // Extra raw extxyz key=value metadata, one line
    std::vector<Property> properties;   // Forces and other per-atom arrays
    std::size_t size() const noexcept { return numbers.size(); }
};

// General property table. info_extxyz may include Lattice and pbc, but not codec controls.
struct RawFrame {
    std::size_t count = 0;
    std::string info_extxyz;
    std::vector<Property> properties;
};

std::string element_symbol(std::uint8_t number);
Frame as_structure(const RawFrame& frame);

enum class StringMode { automatic, fixed, variable };
struct Options {
    std::optional<bool> compressed;     // Unspecified: infer from .zbxyz suffix
    int compression_level = 3;
    double float_rtol = 1e-7;
    double float_atol = 0.0;
    StringMode string_mode = StringMode::automatic;
};

// Owns a binary file. next() verifies all frames read; read(i) seeks past earlier payloads.
class Reader {
public:
    explicit Reader(const std::filesystem::path& path, bool require_checksum = false);
    bool next(Frame& frame);
    bool next_raw(RawFrame& frame);
    RawFrame read_raw(std::size_t index);
    Frame read(std::size_t index);       // Zero-based, resets the cursor and reads one frame
    // Header-only scan: returns this frame's offset and skips its payload (no checksum verification).
    bool scan(std::uint64_t& offset);
    // Offset must come from scan() on this same, unchanged file. index is for error reporting.
    Frame read_at(std::uint64_t offset, std::size_t index = 0);
private:
    bool advance(RawFrame* frame);
    std::ifstream stream_;
    std::uint64_t file_size_ = 0;
    std::size_t index_ = 0;
    bool require_checksum_ = false;
};

// Open once for an MD run; write() never reads previous frames.
// append=true validates the existing file once, during construction.
class Writer {
public:
    explicit Writer(const std::filesystem::path& path, Options options = {}, bool append = false);
    void write(const Frame& frame);
    void write(const RawFrame& frame);
    void flush();
private:
    std::ofstream stream_;
    Options options_;
    bool compressed_ = false;
    std::size_t index_ = 0;
    bool failed_ = false;
};

std::size_t verify(const std::filesystem::path& path); // Requires checksums on every frame

} // namespace binary_xyz
