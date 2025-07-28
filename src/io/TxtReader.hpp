#ifndef VVM_READERS_TXTREADER_HPP
#define VVM_READERS_TXTREADER_HPP

#include "Reader.hpp"
#include <vector>
#include <string>
#include <map>
#include <utility>

namespace VVM {
namespace IO {

class TxtReader : public Reader {
public:
    TxtReader(const std::string& filepath, const VVM::Core::Grid& grid);
    void read_and_initialize(VVM::Core::State& state) override;

private:
    double linear_interpolate(const std::string& field_name, double target_z) const;

    const VVM::Core::Grid& grid_;
    std::string source_file_;
    std::map<std::string, std::vector<std::pair<double, double>>> all_profiles_;
};

} // namespace IO
} // namespace VVM

#endif // VVM_READERS_TXTREADER_HPP
