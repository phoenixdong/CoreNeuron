/*
# =============================================================================
# Copyright (c) 2016 - 2021 Blue Brain Project/EPFL
#
# See top-level LICENSE file for details.
# =============================================================================
*/
#pragma once
#include <cstddef>

#define nrn_pragma_stringify(x) #x
#if defined(CORENEURON_ENABLE_GPU) && defined(CORENRN_PREFER_OPENMP_OFFLOAD) && defined(_OPENMP)
#define nrn_pragma_acc(x)
#define nrn_pragma_omp(x) _Pragma(nrn_pragma_stringify(omp x))
#elif defined(CORENEURON_ENABLE_GPU) && !defined(CORENRN_PREFER_OPENMP_OFFLOAD) && \
    defined(_OPENACC)
#define nrn_pragma_acc(x) _Pragma(nrn_pragma_stringify(acc x))
#define nrn_pragma_omp(x)
#else
#define nrn_pragma_acc(x)
#define nrn_pragma_omp(x)
#endif

namespace coreneuron {
void* cnrn_gpu_copyin(void* h_ptr, std::size_t len);
void cnrn_memcpy_to_device(void* d_ptr, void* h_ptr, std::size_t len);
void cnrn_target_delete(void* h_ptr, std::size_t len);
void* cnrn_target_deviceptr(void* h_ptr);
} // namespace coreneuron
