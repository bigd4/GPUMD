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
    for (const auto& ext : {"bxyz", "zbxyz"}) {
      trajectory::Config cfg; cfg.parse({"output_format", ext, "precision", "lossless"});
      trajectory::Output output; output.configure(cfg);
      trajectory::Frame f; f.count=2;
      f.info="Lattice=\"4 0 0 0 4 0 0 0 4\" pbc=\"T F T\"";
      f.add_strings("species", {"C", "C"}, {0,1});
      f.add_real("pos", 3, std::vector<double>{.1234567890123,1,2,3,4,5}, 2, {0,1});
      auto path=(dir/(std::string("path.")+ext)).string();
      output.write(path,f,false); output.write(path,f,true); output.write(path,f,true);
      output.write(path,f,false); output.write(path,f,true); // overwrite a previously open path
      trajectory::ModelReader reader(path); trajectory::Frame back;
      check(reader.next(back) && back.count==2 && !back.pbc[1]);
      check(back.arrays[1].real==f.arrays[1].real);
      check(reader.next(back) && !reader.next(back));
      output.configure(cfg); output.write(path,f,true);
      check(verify(path)==3);
      auto bad=dir/(std::string("bad_path.")+ext);
      auto bytes=std::filesystem::file_size(path);
      std::filesystem::copy_file(path,bad,std::filesystem::copy_options::overwrite_existing);
      std::filesystem::resize_file(bad,bytes-1);
      trajectory::ModelReader broken(bad.string());
      check(broken.next(back) && broken.next(back));
      rejects([&]{broken.next(back);});
    }
    for (const auto& ext : {"bxyz", "zbxyz"}) {
      trajectory::Config cfg; cfg.parse({"output_format", ext});
      trajectory::Output output; output.configure(cfg);
      trajectory::Frame f; f.count=1;
      // At 100 angstrom, float32 rounding exceeds 1e-6 but satisfies the old rtol.
      const std::vector<double> large{100.000003, 0., 0.};
      const std::vector<double> small{10.123456789, 0., 0.};
      f.add_real("pos",3,large,1,{0});
      f.add_real("unwrapped_position",3,large,1,{0});
      f.add_real("forces",3,large,1,{0});
      auto path=(dir/(std::string("precision.")+ext)).string();
      output.write(path,f,false);
      auto raw=Reader(path).read_raw(0);
      check(std::get<std::vector<double>>(raw.properties[0].values)==large);
      check(std::get<std::vector<double>>(raw.properties[1].values)==large);
      check(std::get<std::vector<double>>(raw.properties[2].values)!=large);
      f.arrays.clear(); f.add_real("pos",3,small,1,{0});
      output.write(path,f,false);
      auto values=std::get<std::vector<double>>(Reader(path).read_raw(0).properties[0].values);
      check(values!=small && std::abs(values[0]-small[0])<=1e-6);
      cfg.lossless=true;output.configure(cfg);output.write(path,f,false);
      check(std::get<std::vector<double>>(Reader(path).read_raw(0).properties[0].values)==small);
    }
    for (const auto& ext : {"bxyz", "zbxyz"}) {
      for (const auto& name : {"forces", "vel"}) {
        const bool force=std::string(name)=="forces";
        double tolerance=force?1e-5:1e-6;
        // Force rounding at 1000 exceeds 1e-5; velocity rounding at 100 exceeds 1e-6.
        const std::vector<double> outside{force?1000.00002:100.000003,0.,0.};
        const std::vector<double> inside{force?100.000003:10.123456789,0.,0.};
        trajectory::Config cfg;cfg.parse({"output_format",ext});
        trajectory::Output output;output.configure(cfg);
        auto path=(dir/(std::string(name)+"_precision."+ext)).string();
        trajectory::Frame f;f.count=1;f.add_real(name,3,outside,1,{0});
        output.write(path,f,false);
        check(std::get<std::vector<double>>(Reader(path).read_raw(0).properties[0].values)==outside);
        f.arrays.clear();f.add_real(name,3,inside,1,{0});output.write(path,f,false);
        auto values=std::get<std::vector<double>>(Reader(path).read_raw(0).properties[0].values);
        check(values!=inside && std::abs(values[0]-inside[0])<=tolerance);
        cfg.lossless=true;output.configure(cfg);output.write(path,f,false);
        check(std::get<std::vector<double>>(Reader(path).read_raw(0).properties[0].values)==inside);
      }
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
