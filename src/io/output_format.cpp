#include "output_format.h"
#include <algorithm>
#include <fstream>
#include <map>
#include <stdexcept>
#ifdef USE_BXYZ
#include <binary_xyz/binary_xyz.hpp>
#endif
namespace trajectory {
void Config::parse(const std::vector<std::string>& t) {
  Config next;
  if(t.size()<2 || (t.size()-2)%2) throw std::runtime_error("output_format requires a format and optional key/value pairs");
  next.format=t[1];
  if(next.format!="xyz" && next.format!="bxyz" && next.format!="zbxyz") throw std::runtime_error("output_format must be xyz, bxyz or zbxyz");
#ifndef USE_BXYZ
  if(next.format!="xyz") throw std::runtime_error("binary output requires a USE_BXYZ=1 build");
#endif
  bool precision=false, level=false;
  for(std::size_t i=2;i<t.size();i+=2) {
    if(t[i]=="precision" && !precision && next.format!="xyz") {
      precision=true;
      if(t[i+1]!="auto" && t[i+1]!="lossless") throw std::runtime_error("precision must be auto or lossless");
      next.lossless=t[i+1]=="lossless";
    } else if(t[i]=="compression_level" && !level && next.format=="zbxyz") {
      level=true;
      if(t[i+1].empty() || t[i+1].find_first_not_of("0123456789")!=std::string::npos) throw std::runtime_error("compression_level must be an integer from 1 to 22");
      auto n=std::stoul(t[i+1]);
      if(n<1 || n>22) throw std::runtime_error("compression_level must be from 1 to 22");
      next.compression_level=static_cast<int>(n);
    } else throw std::runtime_error("unknown, repeated or inapplicable output_format option: "+t[i]);
  }
  *this=next;
}
void Frame::add_strings(const std::string& name, const std::vector<std::string>& values,
                        const std::vector<int>& indices) {
  Array a{name,1,'S',{},{},{}};
  for(int i:indices) a.strings.push_back(values.at(i));
  arrays.push_back(std::move(a));
}
std::string filename(const std::string& original, const Config& config) {
  if(config.format=="xyz") return original;
  auto dot=original.find_last_of('.');
  if(dot!=std::string::npos) {
    auto suffix=original.substr(dot);
    if(suffix==".xyz" || suffix==".bxyz" || suffix==".zbxyz")
      return original.substr(0,dot)+"."+config.format;
  }
  return original+"."+config.format;
}
bool is_binary(const std::string& path) {
  std::ifstream file(path,std::ios::binary);
  std::string first, second;
  std::getline(file,first);std::getline(file,second);
  // Also route files with binary suffixes through the validating binary reader.
  auto dot=path.find_last_of('.');
  auto ext=dot==std::string::npos ? "" : path.substr(dot);
  return second.find("BXYZVersion=")!=std::string::npos || ext==".bxyz" || ext==".zbxyz";
}
struct Output::Impl {
#ifdef USE_BXYZ
  std::map<std::string,std::unique_ptr<binary_xyz::Writer>> writers;
#endif
};
Output::Output():impl_(new Impl) {}
Output::~Output() = default;
void Output::configure(const Config& config) { impl_.reset(new Impl); config_=config; }
void Output::write(const std::string& original, const Frame& frame, bool append) {
#ifdef USE_BXYZ
  if(!enabled()) throw std::runtime_error("binary writer used in text mode");
  binary_xyz::RawFrame raw;raw.count=frame.count;raw.info_extxyz=frame.info;
  for(const auto& a:frame.arrays) {
    if(a.type=='R') raw.properties.push_back({a.name,a.components,a.real});
    else if(a.type=='I') raw.properties.push_back({a.name,a.components,a.integer});
    else if(a.type=='S') raw.properties.push_back({a.name,a.components,a.strings});
    else throw std::runtime_error("unsupported output array type");
  }
  binary_xyz::Options opts;opts.compressed=config_.format=="zbxyz";
  opts.compression_level=config_.compression_level;
  if(config_.lossless) opts.float_rtol=opts.float_atol=0;
  else {
    // Cartesian coordinates are in angstrom: enforce an absolute error bound.
    opts.property_tolerances["pos"] = {0.0, 1e-6};
    opts.property_tolerances["unwrapped_position"] = {0.0, 1e-6};
    opts.property_tolerances["forces"] = {0.0, 1e-5}; // eV/angstrom
    opts.property_tolerances["vel"] = {0.0, 1e-6}; // angstrom/fs
  }
  auto path=filename(original,config_);
  try {
    if(!append) { impl_->writers.erase(path); binary_xyz::Writer writer(path,opts);writer.write(raw);writer.flush(); }
    else {
      auto& writer=impl_->writers[path];
      if(!writer) writer.reset(new binary_xyz::Writer(path,opts,true));
      writer->write(raw);writer->flush();
    }
  } catch(const std::exception& e) {throw std::runtime_error(path+": "+e.what());}
#else
  throw std::runtime_error("binary output requires a USE_BXYZ=1 build");
#endif
}
#ifdef USE_BXYZ
static Frame model_frame(binary_xyz::RawFrame raw) {
  // Discard properties not used for initialization before structured conversion.
  auto model=raw;
  model.properties.erase(std::remove_if(model.properties.begin(),model.properties.end(),[](const binary_xyz::Property& p) {
    return p.name!="species" && p.name!="Z" && p.name!="pos" && p.name!="mass" && p.name!="charge" && p.name!="vel" && p.name!="group";
  }),model.properties.end());
  auto structure=binary_xyz::as_structure(model);
  Frame result;result.count=raw.count;result.info=raw.info_extxyz;
  std::copy(structure.cell.begin(),structure.cell.end(),result.cell);
  std::copy(structure.pbc.begin(),structure.pbc.end(),result.pbc);
  // Normalize Z:I:1 input to symbols using the validated structure interface.
  if(std::none_of(raw.properties.begin(),raw.properties.end(),[](const binary_xyz::Property& p){return p.name=="species";})) {
    std::vector<std::string> labels;
    for(auto z:structure.numbers) labels.push_back(binary_xyz::element_symbol(z));
    raw.properties.push_back({"species",1,std::move(labels)});
  }
  for(const auto& p:raw.properties) {
    Array a{p.name,p.components,'R',{},{},{}};
    if(auto v=std::get_if<std::vector<double>>(&p.values)) a.real=*v;
    else if(auto v=std::get_if<std::vector<std::int64_t>>(&p.values)) {a.type='I';a.integer=*v;}
    else if(auto v=std::get_if<std::vector<std::string>>(&p.values)) {a.type='S';a.strings=*v;}
    else {a.type='L';const auto& flags=std::get<std::vector<std::uint8_t>>(p.values);a.integer.assign(flags.begin(),flags.end());}
    result.arrays.push_back(std::move(a));
  }
  return result;
}
#endif
struct ModelReader::Impl {
#ifdef USE_BXYZ
  binary_xyz::Reader reader;
  explicit Impl(const std::string& path):reader(path) {}
#else
  explicit Impl(const std::string&) { throw std::runtime_error("binary input requires a USE_BXYZ=1 build"); }
#endif
};
ModelReader::ModelReader(const std::string& path):impl_(new Impl(path)) {}
ModelReader::~ModelReader() = default;
bool ModelReader::next(Frame& frame) {
#ifdef USE_BXYZ
  binary_xyz::RawFrame raw;
  if(!impl_->reader.next_raw(raw)) return false;
  frame=model_frame(std::move(raw));
  return true;
#else
  throw std::runtime_error("binary input requires a USE_BXYZ=1 build");
#endif
}
Frame read_model(const std::string& path) {
  ModelReader reader(path);
  Frame frame;
  if(!reader.next(frame)) throw std::runtime_error("empty binary model: "+path);
  return frame;
}
} // namespace trajectory
