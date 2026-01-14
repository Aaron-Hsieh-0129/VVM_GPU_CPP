#ifndef VVM_READERS_TXTREADER_HPP
#define VVM_READERS_TXTREADER_HPP

#include "Reader.hpp"
#include "core/Parameters.hpp"
#include "utils/ConfigurationManager.hpp"
#include <vector>
#include <string>
#include <map>
#include <utility>

namespace VVM {
namespace IO {

class TxtReader : public Reader {
public:
    TxtReader(const std::string& filepath, 
              const VVM::Core::Grid& grid, 
              const VVM::Core::Parameters& params, 
              const VVM::Utils::ConfigurationManager& config);

    void read_and_initialize(VVM::Core::State& state) override;

private:
    const VVM::Core::Grid& grid_;
    const VVM::Core::Parameters& params_;
    const VVM::Utils::ConfigurationManager& config_;
    std::string source_file_;

    std::map<std::string, std::vector<double>> raw_data_;
    std::vector<double> input_z_;

    void read_file();
    void calculate_input_heights();
    void initialize_thermodynamics(VVM::Core::State& state);
    void initialize_forcing(VVM::Core::State& state);
    double interpolate(double target_x, const std::vector<double>& x_vec, const std::vector<double>& y_vec, bool is_pressure_coord = false) const;
};


/*
class TxtReader : public Reader {
public:
    TxtReader(const std::string& filepath, const VVM::Core::Grid& grid, const VVM::Core::Parameters& params, const VVM::Utils::ConfigurationManager& config);
    void read_and_initialize(VVM::Core::State& state) override;
    std::vector<double> zt_input;

private:
    double linear_interpolate(const std::string& field_name, double target_z) const;
    void calculate_input_z();

    const VVM::Utils::ConfigurationManager& config_;
    const VVM::Core::Grid& grid_;
    const VVM::Core::Parameters& params_;
    std::string source_file_;
    std::map<std::string, std::vector<std::pair<double, double>>> all_profiles_;
};
*/

} // namespace IO
} // namespace VVM

#endif // VVM_READERS_TXTREADER_HPP
