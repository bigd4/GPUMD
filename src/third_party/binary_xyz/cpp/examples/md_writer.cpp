#include <binary_xyz/binary_xyz.hpp>
#include <iostream>

int main(int argc,char** argv) {
    if(argc!=2) {std::cerr<<"Usage: bxyz_md_example OUTPUT.zbxyz\n";return 2;}
    try {
        binary_xyz::Options options;
        options.compression_level=3;
        binary_xyz::Writer trajectory(argv[1],options); // Open once, not once per timestep.
        binary_xyz::Frame frame;
        frame.numbers={22,8}; // Ti, O
        frame.positions={0.1,0.2,0.3,1.1,1.2,1.3};
        frame.cell={5,0,0,0,5,0,0,0,5};
        frame.pbc={true,true,true};
        frame.properties.push_back({"forces",3,std::vector<double>{0.1,0,0,-0.1,0,0}});
        for(int step=0;step<3;++step) {
            frame.info_extxyz="step="+std::to_string(step)+" energy=-1.25 comment=\"C++ MD example\"";
            frame.positions[0]=0.1+0.01*step;
            trajectory.write(frame); // Replace the sample update with the MD engine's coordinates.
        }
        trajectory.flush(); // Explicitly report buffered output errors.
        std::cout<<"Wrote 3 independent frames\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 2;}
}
