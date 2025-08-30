#include "Timer.hpp"

namespace VVM {
namespace Utils {

Timer::Timer(const std::string& name) : name_(name) {
    TimingManager::get_instance().start_timer(name_);
}

Timer::~Timer() {
    TimingManager::get_instance().stop_timer(name_);
}

} // namespace Utils
} // namespace VVM
