#ifndef VVM_UTILS_TIMER_HPP
#define VVM_UTILS_TIMER_HPP

#include <string>
#include "TimingManager.hpp"

namespace VVM {
namespace Utils {

class Timer {
public:
    explicit Timer(const std::string& name);
    ~Timer();

private:
    std::string name_;
};

} // namespace Utils
} // namespace VVM

#endif // VVM_UTILS_TIMER_HPP
