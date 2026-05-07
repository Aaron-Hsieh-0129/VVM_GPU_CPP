#ifndef VVM_READERS_TXTREADER_HPP
#define VVM_READERS_TXTREADER_HPP

#include "Reader.hpp"
#include "core/Parameters.hpp"
#include "core/vvm_types.hpp"
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

    std::map<std::string, std::vector<VVM::Real>> raw_data_;
    std::vector<VVM::Real> input_z_;

    void read_file();
    void calculate_input_heights();
    void initialize_thermodynamics(VVM::Core::State& state);
    void initialize_forcing(VVM::Core::State& state);
    VVM::Real interpolate(VVM::Real target_x, const std::vector<VVM::Real>& x_vec, const std::vector<VVM::Real>& y_vec, bool is_pressure_coord = false) const;
};

} // namespace IO
} // namespace VVM

#endif // VVM_READERS_TXTREADER_HPP
