#pragma once

#include <string>
#include <time.h>

#include <generic/common.hpp>

#ifdef __OPENMP__
#include <omp.h>
#endif

namespace Ensign{

/// This timer class measures the elapsed time between two events. Timers can be
/// started and stopped repeatedly. The total time as well as the average time
/// between two events can be queried using the total() and average() methods,
/// respectively.
struct timer {
    timespec t_start;
    bool running;
    double   elapsed;
    unsigned counter;
    double elapsed_sq;

    timer();

    void reset();
    void start();
    double stop();

    double total();
    double average();
    double deviation();
    unsigned count();
};

namespace gt {

    bool is_master();
    void reset();
    void print();

    std::string sorted_output();
    void start(std::string name);
    void stop(std::string name);
    double total(std::string name);
    double average(std::string name);
    double deviation(std::string name);

}

} // namespace Ensign