#pragma once

#include "coreneuron/sim/multicore.hpp"

namespace coreneuron {

void CudaDeliverSetup(const NrnThread* ptrNrnThread);
void CudaDeliverCleanup();

}  // namespace coreneuron
