#include "IOServer.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <map>
#include <algorithm>
#include <adios2.h>
#include <sys/stat.h>

namespace VVM {
namespace IO {

std::string format_six_digits(int number) {
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(6) << number;
    return ss.str();
}

void get_local_range(size_t global_size, int rank, int size, size_t &start, size_t &count) {
    size_t base = global_size / size;
    size_t rem = global_size % size;
    if (rank < rem) { start = rank * (base + 1); count = base + 1; } 
    else { start = rank * base + rem; count = base; }
}

void run_io_server(MPI_Comm io_comm, const VVM::Utils::ConfigurationManager& config) {
    int rank, size;
    MPI_Comm_rank(io_comm, &rank);
    MPI_Comm_size(io_comm, &size);

    adios2::ADIOS adios(io_comm);
    std::string output_dir = config.get_value<std::string>("output.output_dir");
    std::string filename_prefix = config.get_value<std::string>("output.output_filename_prefix");
    
    std::string input_stream_name = filename_prefix;

    if (rank == 0) {
        std::cout << "  [IO-Server] Listening on stream: " << input_stream_name << std::endl;
        mkdir(output_dir.c_str(), 0777);
    }

    adios2::IO inIO = adios.DeclareIO("InputSST");
    inIO.SetEngine("SST");

    adios2::IO outIO = adios.DeclareIO("OutputHDF5");
    outIO.SetEngine("HDF5"); 
    outIO.SetParameter("IdleH5Writer", "true"); 
    
    if (size > 1) {
        outIO.SetParameter("H5CollectiveMPIO", "true"); 
        if (rank == 0) std::cout << "  [IO-Server] HDF5 Collective Mode: ENABLED (Ranks: " << size << ")" << std::endl;
    } 
    else {
        outIO.SetParameter("H5CollectiveMPIO", "false");
        if (rank == 0) std::cout << "  [IO-Server] HDF5 Collective Mode: DISABLED (Single Rank)" << std::endl;
    }

    adios2::Engine reader = inIO.Open(input_stream_name, adios2::Mode::Read);
    adios2::Engine writer;

    while (true) {
        adios2::StepStatus status = reader.BeginStep();
        if (status != adios2::StepStatus::OK) break;

        int step = reader.CurrentStep();
        const auto& varTypeMap = inIO.AvailableVariables();
        std::map<std::string, std::vector<double>> data_buffers;
        std::vector<std::string> current_step_vars; 

        for (const auto& varPair : varTypeMap) {
            std::string name = varPair.first;
            std::string type = varPair.second.at("Type");
            if (type != "double") continue;

            auto varIn = inIO.InquireVariable<double>(name);
            auto shape = varIn.Shape();
            current_step_vars.push_back(name);

            if (!outIO.InquireVariable<double>(name)) {
                outIO.DefineVariable<double>(name, shape);
            }

            size_t my_start, my_count;
            if (shape.size() > 0) {
                get_local_range(shape[0], rank, size, my_start, my_count);
            } 
            else {
                my_start = 0; my_count = 0;
            }

            if (my_count > 0) {
                adios2::Dims start = varIn.Start();
                adios2::Dims count = varIn.Shape();
                start[0] = my_start;
                count[0] = my_count;
                varIn.SetSelection({start, count});

                size_t elements = 1;
                for(auto c : count) elements *= c;
                data_buffers[name].resize(elements);
                reader.Get(varIn, data_buffers[name].data());
            }
        }
        reader.EndStep();

        if (rank == 0) std::cout << "  [IO-Server] Writing Step " << step << "..." << std::endl;
        std::string h5_name = output_dir + "/" + filename_prefix + "_" + format_six_digits(step) + ".h5";
        writer = outIO.Open(h5_name, adios2::Mode::Write, io_comm);
        writer.BeginStep();

        std::sort(current_step_vars.begin(), current_step_vars.end());

        for (const auto& name : current_step_vars) {
            auto varOut = outIO.InquireVariable<double>(name);
            if (!varOut) continue;

            if (data_buffers.count(name)) {
                size_t my_count = data_buffers[name].size();
                size_t dim0 = varOut.Shape()[0];
                size_t s_start, s_count;
                get_local_range(dim0, rank, size, s_start, s_count);
                
                if (s_count > 0) {
                     adios2::Dims start = {s_start};
                     adios2::Dims count = {s_count};
                     // Fill rest of dims
                     for(size_t i=1; i<varOut.Shape().size(); ++i) {
                         start.push_back(0);
                         count.push_back(varOut.Shape()[i]);
                     }
                     varOut.SetSelection({start, count});
                     writer.Put(varOut, data_buffers[name].data());
                }
            }
        }
        writer.EndStep();
        writer.Close();
    }
    reader.Close();
}

} // namespace IO
} // namespace VVM
