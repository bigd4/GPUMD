#include <binary_xyz/binary_xyz.hpp>
#include <chrono>
#include <iostream>

int main(int argc,char** argv) {
    using namespace binary_xyz;
    try {
        if(argc>=3 && std::string(argv[1])=="verify") {
            std::cout<<"Verified "<<verify(argv[2])<<" frames (SHA-256)\n";return 0;
        }
        if(argc>=3 && std::string(argv[1])=="inspect") {
            Reader reader(argv[2]);
            if(argc==4) {
                std::string index=argv[3];
                if(index.empty()||index.find_first_not_of("0123456789")!=std::string::npos) throw Error("index must be nonnegative");
                auto frame=reader.read(std::stoull(index));
                std::cout<<"frame="<<index<<" atoms="<<frame.size()<<" properties="<<frame.properties.size()<<'\n';
            } else {Frame frame;std::size_t n=0,total=0;while(reader.next(frame)){++n;total+=frame.size();}
                std::cout<<"frames="<<n<<" atoms="<<total<<'\n';}
            return 0;
        }
        if((argc==4||argc==5) && std::string(argv[1])=="convert") {
            const std::filesystem::path input=argv[2],output=argv[3];
            if(std::filesystem::exists(output)) throw Error("output already exists");
            Options options;
            options.compressed=output.extension()==".zbxyz";
            if(output.extension()!=".bxyz"&&output.extension()!=".zbxyz") throw Error("output must be .bxyz or .zbxyz");
            if(argc==5) options.compression_level=std::stoi(argv[4]);
            options.float_rtol=0; // Converting existing binary data never reduces precision further.
            auto temporary=output;
            temporary+=".tmp."+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
            try {
                {Reader reader(input);Writer writer(temporary,options);Frame frame;
                    while(reader.next(frame)) writer.write(frame);writer.flush();}
                // Same-directory hard link publishes without overwriting a concurrently created file.
                std::filesystem::create_hard_link(temporary,output);
                std::filesystem::remove(temporary);
            } catch(...) {std::error_code ec;std::filesystem::remove(temporary,ec);throw;}
            return 0;
        }
        std::cerr<<"Usage: bxyz_cpp verify FILE | inspect FILE [INDEX] | convert INPUT OUTPUT [LEVEL]\n";
        return 2;
    } catch(const std::exception& e) {std::cerr<<"bxyz_cpp: "<<e.what()<<'\n';return 2;}
}
