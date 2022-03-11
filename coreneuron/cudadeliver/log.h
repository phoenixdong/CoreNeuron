#pragma once

#include "coreneuron/mpi/nrnmpi.h"
#include "coreneuron/sim/multicore.hpp"

namespace coreneuron {

// #define ENABLE_LOGGING_TRACE 1
#define ENABLE_LOGGING_INFO 1

#define __LOG_INTERNAL(_Level_, _Msg_, ...) \
    do { \
        fprintf(stderr, "[%08.3lf] %-5s #%03d#[%s:%d] " _Msg_ "\n", \
            nrn_threads[0]._t, \
            (_Level_), \
            nrnmpi_myid, \
            (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__), \
            __LINE__, \
            ##__VA_ARGS__); \
        fflush(stderr); \
    } while(false)


#if ENABLE_LOGGING_TRACE
#define TRACE(_Msg_, ...)       __LOG_INTERNAL("TRACE", _Msg_, ##__VA_ARGS__)
#else
#define TRACE(_Msg_, ...)
#endif

#if ENABLE_LOGGING_INFO
#define INFO(_Msg_, ...)         __LOG_INTERNAL("INFO", _Msg_, ##__VA_ARGS__)
#else
#define INFO(_Msg_, ...)
#endif
}  // namespace coreneuron
