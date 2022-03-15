#pragma once

#include "coreneuron/sim/multicore.hpp"

//dong
//#include <mpi.h>
#include </usr/lib/x86_64-linux-gnu/openmpi/include/mpi.h>

namespace coreneuron {
extern MPI_Comm multisendComm;
extern int multiSendCnt;
extern int multiRecvCnt;

extern int maxSpikeSize;
extern int* cFiredPreSynIds;
extern int recvSpikeSize;
extern int* cRecvSpikeGids;
extern int* gRecvSpikeGids;
extern double* cRecvSpikeTimes;
extern double* gRecvSpikeTimes;

extern int maxFiredNidSize;
extern int* gPtrFiredNidSize;
extern int* cFiredNids;
extern int* gFiredNids;
extern double* cFiredNidTimes;
extern double* gFiredNidTimes;

void CudaDeliver(const int* gFiredPreSynIds, const int size, NrnThread* ptrNrnThread);
void CudaFire(NrnThread* ptrNrnThread);
void CudaIncHalfTimeStep(const NrnThread* ptrNrnThread);
void CudaDeliverSyncronize(const NrnThread* ptrNrnThread);
void CudaDeliverCommitCpuRecvSpikesToFiredTable(const NRNMPI_Spike* spikes, const int size, const NrnThread* ptrNrnThread);

}  // namespace coreneuron
