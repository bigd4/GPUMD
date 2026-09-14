#include "io/output_format.h"
#include <binary_xyz/binary_xyz.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cmath>
using namespace binary_xyz;
void check(bool b) { if(!b) throw std::runtime_error("check failed"); }
template<class F> void rejects(F f) {try {f();}catch(const std::exception&) {return;}throw std::runtime_error("expected rejection");}
int main(int argc,char** argv) {
  try {
    std::filesystem::path dir=argv[1];std::filesystem::create_directories(dir);
    for(const auto& ext:{"bxyz","zbxyz"}) {
      auto path=dir/(std::string("raw.")+ext);
      RawFrame f;f.count=2;f.info_extxyz="Time=1 virial=\"1 2 3 4 5 6 7 8 9\"";
      f.properties={{"vel",3,std::vector<double>{.1,-0.,2,3,4,5}},
                    {"virial",9,std::vector<double>(18,1.234567890123)},
                    {"label",1,std::vector<std::string>{"CG_A",u8"粒子"}}};
      Options lossless;lossless.float_rtol=0;
      {Writer w(path,lossless);w.write(f);w.flush();}
      auto back=Reader(path).read_raw(0);
      check(back.count==2 && back.properties.size()==3);
      check(std::get<std::vector<double>>(back.properties[0].values)==std::get<std::vector<double>>(f.properties[0].values));
      check(std::signbit(std::get<std::vector<double>>(back.properties[0].values)[1]));
      rejects([&]{Reader(path).read(0);});
      {Writer w(path,lossless,true);w.write(f);w.flush();}
      check(verify(path)==2);
      RawFrame empty;empty.properties={{"vel",3,std::vector<double>{}}};
      {Writer w(dir/(std::string("empty.")+ext));w.write(empty);w.flush();}
      check(Reader(dir/(std::string("empty.")+ext)).read_raw(0).count==0);
      RawFrame cg;cg.count=2;cg.info_extxyz="Lattice=\"4 0 0 0 4 0 0 0 4\" pbc=\"T T T\"";
      cg.properties={{"species",1,std::vector<std::string>{"CG_A","CG_B"}}, {"pos",3,std::vector<double>(6)}};
      {Writer w(dir/(std::string("cg.")+ext));w.write(cg);w.flush();}
      rejects([&]{Reader(dir/(std::string("cg.")+ext)).read(0);});
      auto broken=dir/(std::string("broken.")+ext);
      std::filesystem::copy_file(path,broken,std::filesystem::copy_options::overwrite_existing);
      {std::fstream file(broken,std::ios::in|std::ios::out|std::ios::binary);file.seekp(-1,std::ios::end);file.put('!');}
      rejects([&]{verify(broken);});
      rejects([&]{Writer w(broken,{},true);});
      f.properties[0].components=4;
      rejects([&]{Writer w(dir/(std::string("invalid.")+ext));w.write(f);});
    }
    trajectory::Config config;
    config.parse({"output_format","zbxyz","compression_level","6","precision","lossless"});
    check(config.lossless && config.compression_level==6);
    config.parse({"output_format","bxyz"});check(!config.lossless && config.compression_level==3);
    for(auto t:std::vector<std::vector<std::string>>{
      {"output_format"},{"output_format","bad"},{"output_format","xyz","precision","auto"},
      {"output_format","bxyz","compression_level","3"},{"output_format","zbxyz","compression_level","0"},
      {"output_format","zbxyz","compression_level","23"},{"output_format","zbxyz","precision"},
      {"output_format","bxyz","precision","auto","precision","lossless"}}) rejects([&]{config.parse(t);});
    check(trajectory::filename("movie.xyz",config)=="movie.bxyz");
    check(trajectory::filename("custom.42",config)=="custom.42.bxyz");
    std::cout<<"raw codec and configuration checks passed\n";
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
