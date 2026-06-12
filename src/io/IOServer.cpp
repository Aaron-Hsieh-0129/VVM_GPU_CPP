#include "IOServer.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <map>
#include <algorithm>
#include <cmath>
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

    const std::string output_dir =
        config.get_value<std::string>("output.output_dir");
    const std::string filename_prefix =
        config.get_value<std::string>("output.output_filename_prefix");
    const VVM::Real output_interval_s =
        config.get_value<VVM::Real>("simulation.output_interval_s");

    const std::string input_stream_name = output_dir + "/" + filename_prefix;

    if (rank == 0) {
        std::cout << "  [IO-Server] Listening on stream: "
                  << input_stream_name << std::endl;
        mkdir(output_dir.c_str(), 0777);
    }

    MPI_Barrier(io_comm);

    // -------------------------
    // SST reader: safe settings
    // -------------------------
    adios2::IO inIO = adios.DeclareIO("InputSST");
    inIO.SetEngine("SST");

    // Force the same SST path every time. Do not let ADIOS2 auto-select RDMA.
    inIO.SetParameter("DataTransport", "WAN");
    inIO.SetParameter("WANDataTransport", "sockets");
    inIO.SetParameter("ControlTransport", "sockets");

    // Restart startup can be long before the writer opens.
    inIO.SetParameter("OpenTimeoutSecs", "14400");
    inIO.SetParameter("SpeculativePreloadMode", "OFF");
    inIO.SetParameter("AlwaysProvideLatestTimestep", "false");
    
    // -------------------------
    // HDF5 writer: safe settings
    // -------------------------
    adios2::IO outIO = adios.DeclareIO("OutputHDF5");
    outIO.SetEngine("HDF5");
    outIO.SetParameter("IdleH5Writer", "true");

    // For the safe SST run, use 1 IO rank first.
    // Serial HDF5 is safer than collective HDF5 while debugging SST stability.
    outIO.SetParameter("H5CollectiveMPIO", "false");

    adios2::Engine reader;
    try {
        reader = inIO.Open(input_stream_name, adios2::Mode::Read);
    } catch (const std::exception& e) {
        if (rank == 0) {
            std::cerr << "[IO-Server] FATAL: failed to open SST stream: "
                      << e.what() << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 10);
    }

    while (true) {
        std::map<std::string, std::vector<VVM::Real>> data_buffers;
        std::vector<std::string> current_step_vars;

        adios2::StepStatus status;
        try {
            status = reader.BeginStep();
        } catch (const std::exception& e) {
            if (rank == 0) {
                std::cerr << "[IO-Server] FATAL: BeginStep failed: "
                          << e.what() << std::endl;
            }
            MPI_Abort(MPI_COMM_WORLD, 11);
        }

        if (status != adios2::StepStatus::OK) {
            if (rank == 0) {
                std::cout << "[IO-Server] SST stream ended." << std::endl;
            }
            break;
        }

        const int step = reader.CurrentStep();
        VVM::Real step_time = static_cast<VVM::Real>(step) * output_interval_s;

        const auto& varTypeMap = inIO.AvailableVariables();

        try {
            for (const auto& varPair : varTypeMap) {
                const std::string& name = varPair.first;

                auto typeIt = varPair.second.find("Type");
                if (typeIt == varPair.second.end()) continue;

                const std::string& type = typeIt->second;
                if (type != "double" && type != "float") continue;

                auto varIn = inIO.InquireVariable<VVM::Real>(name);
                if (!varIn) continue;

                const adios2::Dims shape = varIn.Shape();
                current_step_vars.push_back(name);

                if (!outIO.InquireVariable<VVM::Real>(name)) {
                    if (shape.empty()) {
                        outIO.DefineVariable<VVM::Real>(name);
                    } else {
                        adios2::Dims start(shape.size(), 0);
                        adios2::Dims count = shape;
                        outIO.DefineVariable<VVM::Real>(name, shape, start, count);
                    }
                }

                // Scalar
                if (shape.empty()) {
                    data_buffers[name].resize(1);

                    // Safe mode: synchronous get. Avoid large deferred PerformGets bursts.
                    reader.Get(varIn, data_buffers[name].data(), adios2::Mode::Sync);

                    if (name == "time") {
                        step_time = data_buffers[name][0];
                    }
                    continue;
                }

                // Array: split first dimension across IO ranks.
                size_t my_start = 0;
                size_t my_count = 0;
                get_local_range(shape[0], rank, size, my_start, my_count);

                if (my_count == 0) {
                    continue;
                }

                adios2::Dims start(shape.size(), 0);
                adios2::Dims count = shape;
                start[0] = my_start;
                count[0] = my_count;

                varIn.SetSelection({start, count});

                size_t elements = 1;
                for (const auto c : count) {
                    elements *= c;
                }

                data_buffers[name].resize(elements);

                // Safe mode: synchronous get.
                reader.Get(varIn, data_buffers[name].data(), adios2::Mode::Sync);
            }

            reader.EndStep();

        } catch (const std::exception& e) {
            if (rank == 0) {
                std::cerr << "[IO-Server] FATAL: SST read/Get/EndStep failed: "
                          << e.what() << std::endl;
            }
            MPI_Abort(MPI_COMM_WORLD, 12);
        }

        const int output_index =
            static_cast<int>(std::llround(step_time / output_interval_s));

        if (rank == 0) {
            std::cout << "  [IO-Server] Writing Step "
                      << output_index << "..." << std::endl;
        }

        const std::string h5_name =
            output_dir + "/" + filename_prefix + "_" +
            format_six_digits(output_index) + ".h5";

        std::sort(current_step_vars.begin(), current_step_vars.end());

        try {
            adios2::Engine writer = outIO.Open(h5_name, adios2::Mode::Write, io_comm);
            writer.BeginStep();

            for (const auto& name : current_step_vars) {
                auto varOut = outIO.InquireVariable<VVM::Real>(name);
                if (!varOut) continue;

                auto bufIt = data_buffers.find(name);
                if (bufIt == data_buffers.end()) continue;

                auto& buffer = bufIt->second;

                // Scalar
                if (varOut.Shape().empty()) {
                    writer.Put(varOut, buffer.data(), adios2::Mode::Sync);
                    continue;
                }

                const adios2::Dims shape = varOut.Shape();
                if (shape.empty()) continue;

                size_t s_start = 0;
                size_t s_count = 0;
                get_local_range(shape[0], rank, size, s_start, s_count);

                if (s_count == 0) {
                    continue;
                }

                adios2::Dims start(shape.size(), 0);
                adios2::Dims count = shape;
                start[0] = s_start;
                count[0] = s_count;

                varOut.SetSelection({start, count});

                writer.Put(varOut, buffer.data(), adios2::Mode::Sync);
            }

            writer.EndStep();
            writer.Close();

        } catch (const std::exception& e) {
            if (rank == 0) {
                std::cerr << "[IO-Server] FATAL: HDF5 write failed for "
                          << h5_name << ": " << e.what() << std::endl;
            }
            MPI_Abort(MPI_COMM_WORLD, 13);
        }
    }

    try {
        reader.Close();
    } catch (const std::exception& e) {
        if (rank == 0) {
            std::cerr << "[IO-Server] Warning: reader.Close() failed: "
                      << e.what() << std::endl;
        }
    }
}

} // namespace IO
} // namespace VVM
