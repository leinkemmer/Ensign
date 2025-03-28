#include <generic/common.hpp>
#include <generic/timer.hpp>

#include <cmath>
#include <map>
#include <iostream>
#include <iomanip>
#include <set>
#include <functional>

#include <sys/time.h>
#include <unistd.h>
#include <string.h>
#include <time.h>

namespace Ensign{

timer::timer() {
    counter = 0;
    elapsed = 0.0;
    running = false;
    elapsed_sq = 0.0;
}

void timer::reset() {
    counter = 0;
    elapsed = 0.0;
    running = false;
    elapsed_sq = 0.0;
}

void timer::start() {
    clock_gettime(CLOCK_MONOTONIC, &t_start);
    running = true;
}

/// The stop method returns the elapsed time since the last call of start().
double timer::stop() {
    if(running == false) {
        cout << "WARNING: timer::stop() has been called without calling "
            << "timer::start() first." << endl;
        return 0.0;
    } else {
        timespec t_end;
        clock_gettime(CLOCK_MONOTONIC, &t_end);
        int sec  = t_end.tv_sec-t_start.tv_sec;
        double nsec = ((double)(t_end.tv_nsec-t_start.tv_nsec));
        if(nsec < 0.0) {
            nsec += 1e9;
            sec--;
        }
        double t = (double)sec + nsec/1e9;
        counter++;
        elapsed += t;
        elapsed_sq += t*t;
        return t;
    }
}

double timer::total() {
    return elapsed;
}

double timer::average() {
    return elapsed/double(counter);
}

double timer::deviation() {
    if (counter == 1) {
        return 0.0;
    }
    else {
        return sqrt(elapsed_sq/double(counter)-average()*average());
    }
}

unsigned timer::count() {
    return counter;
}

namespace gt {
    std::map<std::string,timer> timers;

    bool is_master() {
        #ifdef __OPENMP__
        if(omp_get_thread_num() != 0)
            return false;
        #endif

        return true;
    }

    void reset() {
        for(auto& el : timers)
            el.second.reset();
    }

    void print() {
        for(auto el : timers)
            cout << "gt " << el.first << ": " << el.second.total() << " s"
                << endl;
    }

    std::string sorted_output() {
        typedef std::pair<std::string,timer> pair_nt;
        auto comp = [](pair_nt a1, pair_nt a2) {
            return a1.second.total() > a2.second.total();
        };
        std::set<pair_nt, decltype(comp)> sorted(begin(timers), end(timers), comp);

        std::stringstream ss;
        ss.precision(4);
        ss.setf(std::ios_base::scientific);
        for(auto el : sorted) {
            timer& t = el.second;
            ss << std::setw(40) << el.first
                << std::setw(15) << t.total()
                << std::setw(15) << t.count()
                << std::setw(15) << t.average()
                << std::setw(15) << t.deviation()/t.average() << endl;
        }
        return ss.str();
    }

    void start(std::string name) {
        if(is_master())
            timers[name].start();

    }

    void stop(std::string name) {
        if(is_master())
            timers[name].stop();
    }

    double total(std::string name) {
        return timers[name].total();
    }

    double average(std::string name) {
        return timers[name].average();
    }

    double deviation(std::string name) {
        return timers[name].deviation();
    }
}

} // namespace Ensign