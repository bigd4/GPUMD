#include <binary_xyz/binary_xyz.hpp>
#include <cmath>
#include <iostream>
#include <limits>

using namespace binary_xyz;
void check(bool ok,const char* message) {if(!ok)throw std::runtime_error(message);}
template<class F> void rejects(F action) {
    try {action();}catch(const Error&){return;}
    throw std::runtime_error("expected binary_xyz::Error");
}
int main(int argc,char** argv) {
    try {
        if(argc!=2) return 2;
        auto directory=std::filesystem::path(argv[1]);std::filesystem::create_directories(directory);
        auto path=directory/"frames.zbxyz";
        Frame frame;frame.numbers={22,8};frame.positions={.1,0,0,1.5,-0.0,2};
        frame.properties.push_back({"labels",1,std::vector<std::string>{"",u8"氧气"}});
        frame.properties.push_back({"logical",1,std::vector<std::uint8_t>{0,1}});
        frame.properties.push_back({"integer",1,std::vector<std::int64_t>{INT64_MIN,INT64_MAX}});
        frame.properties.push_back({"real",1,std::vector<double>{1e-300,std::numeric_limits<double>::quiet_NaN()}});
        {Writer writer(path);writer.write(frame);writer.flush();writer.write(Frame{});writer.flush();}
        check(verify(path)==2,"verify frame count");
        auto result=Reader(path).read(0);
        check(result.numbers==frame.numbers,"atomic numbers");
        check(result.positions[0]==static_cast<double>(static_cast<float>(.1)),"default float32 selection");
        check(std::signbit(result.positions[4]),"signed zero");
        check(std::get<std::vector<std::int64_t>>(result.properties[2].values)==
              std::get<std::vector<std::int64_t>>(frame.properties[2].values),"int64 limits");
        check(std::get<std::vector<double>>(result.properties[3].values)[0]==1e-300,"small floats retained");
        check(std::isnan(std::get<std::vector<double>>(result.properties[3].values)[1]),"nan retained");
        {Writer writer(path,{},true);writer.write(frame);writer.flush();}
        check(verify(path)==3,"append preserves prior frames");
        {
            Reader reader(path);std::vector<std::uint64_t> offsets;std::uint64_t offset;
            while(reader.scan(offset)) offsets.push_back(offset);
            check(offsets.size()==3 && offsets[0]==0 && offsets[2]>offsets[1],"header scan offsets");
            check(!reader.scan(offset),"scan repeated EOF");
            check(reader.read_at(offsets[2],2).numbers==frame.numbers,"direct last frame");
            check(reader.read_at(offsets[1],1).size()==0,"direct empty frame");
            Frame next;check(reader.next(next) && next.numbers==frame.numbers,"next after read_at");
            rejects([&]{reader.read_at(std::filesystem::file_size(path));});
            rejects([&]{reader.read_at(1);});
        }
        rejects([&]{Reader(path).read(3);});
        Options strict;strict.float_rtol=0;
        auto lossless=directory/"strict.bxyz";
        {Writer writer(lossless,strict);writer.write(frame);writer.flush();}
        check(Reader(lossless).read(0).positions==frame.positions,"strict double preservation");
        for(auto mode:{StringMode::automatic,StringMode::fixed,StringMode::variable}) {
            Options options;options.string_mode=mode;
            {Writer writer(directory/"strings.zbxyz",options);writer.write(frame);writer.flush();}
            auto f=Reader(directory/"strings.zbxyz").read(0);
            check(std::get<std::vector<std::string>>(f.properties[0].values)==std::get<std::vector<std::string>>(frame.properties[0].values),"string mode");
        }
        auto invalid=directory/"invalid.bxyz";
        {Writer writer(invalid);
            Frame bad=frame;bad.positions.pop_back();rejects([&]{writer.write(bad);});
            bad=frame;bad.numbers[0]=255;rejects([&]{writer.write(bad);});
            bad=frame;bad.info_extxyz="BXYZChecksum=bad";rejects([&]{writer.write(bad);});
            bad=frame;bad.properties.push_back({"pos",3,std::vector<double>(6)});rejects([&]{writer.write(bad);});
            bad=frame;bad.properties.push_back({"badbool",1,std::vector<std::uint8_t>{2,0}});rejects([&]{writer.write(bad);});
            bad=frame;bad.properties.push_back({"badutf8",1,std::vector<std::string>{std::string(1,static_cast<char>(255)),""}});rejects([&]{writer.write(bad);});
            writer.write(frame);writer.flush();}
        check(verify(invalid)==1,"failed validation does not write partial frames");
        auto before=std::filesystem::file_size(path);
        Options bad;bad.float_rtol=-1;rejects([&]{Writer w(path,bad);});
        check(std::filesystem::file_size(path)==before,"invalid options do not truncate");
        {std::fstream file(path,std::ios::binary|std::ios::in|std::ios::out);file.seekp(-1,std::ios::end);char c='x';file.write(&c,1);}
        {
            Reader reader(path);std::uint64_t offset=0,last=0;std::size_t count=0;
            while(reader.scan(offset)) {last=offset;++count;}
            check(count==3,"scan skips corrupt payload without decoding");
            check(reader.read_at(0).numbers==frame.numbers,"unaffected frame still readable");
            rejects([&]{reader.read_at(last,2);});
        }
        rejects([&]{Writer w(path,{},true);});
        check(std::filesystem::file_size(path)==before,"bad append preserves file");
        std::cout<<"C++ API checks passed\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
