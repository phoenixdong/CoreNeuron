/*
# =============================================================================
# Copyright (c) 2016 - 2021 Blue Brain Project/EPFL
#
# See top-level LICENSE file for details.
# =============================================================================
*/
#pragma once
#ifdef _OPENACC
#include <openacc.h>

#define nrn_pragma_stringify(x) #x
// For now we do not support OpenMP offload without OpenACC, that will come soon
#ifdef CORENRN_PREFER_OPENMP_OFFLOAD
#define nrn_pragma_omp(x) _Pragma(nrn_pragma_stringify(omp x))
#define nrn_pragma_acc(x)
#else
#define nrn_pragma_acc(x) _Pragma(nrn_pragma_stringify(acc x))
#define nrn_pragma_omp(x)
#endif
#else
// No OpenACC -> no GPU offload, for now.
#ifdef CORENRN_PREFER_OPENMP_OFFLOAD
#error "Pure OpenMP offload not supported yet"
#endif
#define nrn_pragma_acc(x)
#define nrn_pragma_omp(x)
#endif
