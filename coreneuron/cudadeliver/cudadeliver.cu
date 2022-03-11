#include "cudadeliver.h"
#include "util.h"
#include "firedtable.h"
#include "log.h"

#include "coreneuron/nrniv/nrniv_decl.h"
#include "coreneuron/coreneuron.hpp"
#include "coreneuron/network/multisend.hpp"
#include "coreneuron/utils/profile/profiler_interface.h"
#include "coreneuron/io/output_spikes.hpp"

#include <stdio.h>
namespace coreneuron {

#define PP2NT(pp) (nrn_threads + (pp)->_tid)

const double teps = 1e-10;

MPI_Comm multisendComm;
int multiSendCnt = 0;
int multiRecvCnt = 0;

int maxSpikeSize = 0;
int* cFiredPreSynIds = nullptr;
int recvSpikeSize = 0;
int* cRecvSpikeGids = nullptr;
int* gRecvSpikeGids = nullptr;
double* cRecvSpikeTimes = nullptr;
double* gRecvSpikeTimes = nullptr;

int maxFiredNidSize = 0;
int* gPtrFiredNidSize = nullptr;
int* cFiredNids = nullptr;
int* gFiredNids = nullptr;
double* cFiredNidTimes = nullptr;
double* gFiredNidTimes = nullptr;

void CudaDeliver(const int* gFiredPreSynIds, const int size, NrnThread* ptrNrnThread);
void CudaFire(NrnThread* ptrNrnThread);
void CudaIncHalfTimeStep(const NrnThread* ptrNrnThread);
void CudaDeliverSyncronize(const NrnThread* ptrNrnThread);
void CudaDeliverCommitCpuRecvSpikesToFiredTable(const NRNMPI_Spike* spikes, const int size, const NrnThread* ptrNrnThread);
void fireNetCons(NrnThread* ptrNrnThread, const double lowerBoundFactor, const double upperBoundFactor);
void recordSpikes(const int* preSynIds, const int size, const NrnThread* ptrNrnThread);
void cpuMultiSend(const int* cFiredPreSynIds, const int size, const NrnThread* ptrNrnThread);
void cpuMultiRecv(int* cRecvSpikeGidBuffer, double* cRecvSpikeTimeBuffer, int* ptrSize, const int capacity);
void cpuOutputEvent(const int* cFiredPreSynIds, const int size, const NrnThread* ptrNrnThread);
void commitCpuRecvSpikesToFiredTable(int* cRecvSpikeGidBuffer, double* cRecvSpikeTimeBuffer, int* ptrRecvSpikeSize);

void CudaDeliver(const int* gFiredPreSynIds, const int size, NrnThread* ptrNrnThread) {
    if (ptrNrnThread->id != 0) {
        return;
    }
    Instrumentor::phase p("CudaDeliver");
    const FiredTable& firedTable = *ptrFiredTable;
    assert(firedTable.currTime > (ptrNrnThread->_t - 0.1*ptrNrnThread->_dt));
    assert(firedTable.currTime < (ptrNrnThread->_t + 0.1*ptrNrnThread->_dt));
    if (size > 0) {
        assert(size <= maxSpikeSize);
        CommitLidsToFiredTable(gFiredPreSynIds, size);
        {
        Instrumentor::phase p("MemTransferBeforeMultiSend");
        gpuErrchk(cudaMemcpy(
            (void*)cFiredPreSynIds,
            (void*)gFiredPreSynIds,
            size*sizeof(cFiredPreSynIds[0]),
            cudaMemcpyDeviceToHost
        ));
        }
        recordSpikes(cFiredPreSynIds, size, ptrNrnThread);
#if NRN_MULTISEND
        if (use_multisend_) {
            cpuMultiSend(cFiredPreSynIds, size, ptrNrnThread);
        }
        else
#endif
        {
            cpuOutputEvent(cFiredPreSynIds, size, ptrNrnThread);
        }
    }

#if NRN_MULTISEND
    if (use_multisend_) {
        cpuMultiRecv(cRecvSpikeGids, cRecvSpikeTimes, &recvSpikeSize, maxSpikeSize);
        commitCpuRecvSpikesToFiredTable(cRecvSpikeGids, cRecvSpikeTimes, &recvSpikeSize);
    }
#endif

    fireNetCons(ptrNrnThread, 0., 0.5);

    // IncFiredTableTimeStep();
    // IncFiredTableHalfTimeStep();
}

void CudaFire(NrnThread* ptrNrnThread) {
    if (ptrNrnThread->id != 0) {
        return;
    }
    Instrumentor::phase p("CudaFire");
    const FiredTable& firedTable = *ptrFiredTable;
    assert(firedTable.currTime > (ptrNrnThread->_t - 0.1*ptrNrnThread->_dt));
    assert(firedTable.currTime < (ptrNrnThread->_t + 0.1*ptrNrnThread->_dt));
    fireNetCons(ptrNrnThread, -0.5, 0.);
}

void CudaIncHalfTimeStep(const NrnThread* ptrNrnThread) {
    if (ptrNrnThread->id != 0) {
        return;
    }
    Instrumentor::phase p("CudaIncHalfTimeStep");
    IncFiredTableHalfTimeStep();
}

void CudaDeliverSyncronize(const NrnThread* ptrNrnThread) {
    Instrumentor::phase p("CudaDeliverSyncronize");
    assert(ptrNrnThread->id == 0);
    FiredTable& firedTable = *ptrFiredTable;
    assert(firedTable.currTime > (ptrNrnThread->_t - 0.1*ptrNrnThread->_dt));
    assert(firedTable.currTime < (ptrNrnThread->_t + 0.1*ptrNrnThread->_dt));
    cpuMultiRecv(cRecvSpikeGids, cRecvSpikeTimes, &recvSpikeSize, maxSpikeSize);
    nrnmpi_barrier();
    cpuMultiRecv(cRecvSpikeGids, cRecvSpikeTimes, &recvSpikeSize, maxSpikeSize);
    int ncons = 0;
    while (nrnmpi_multisend_conserve(multiSendCnt, multiRecvCnt, &multisendComm) != 0) {
        cpuMultiRecv(cRecvSpikeGids, cRecvSpikeTimes, &recvSpikeSize, maxSpikeSize);
        ++ncons;
    }
    TRACE("CudaDeliverSyncronize ncons = %d", ncons);
    commitCpuRecvSpikesToFiredTable(cRecvSpikeGids, cRecvSpikeTimes, &recvSpikeSize);
}

void CudaDeliverCommitCpuRecvSpikesToFiredTable(const NRNMPI_Spike* spikes, const int size, const NrnThread* ptrNrnThread) {
    Instrumentor::phase p("CudaDeliverCommitCpuRecvSpikesToFiredTable");
    assert(ptrNrnThread->id == 0);
    FiredTable& firedTable = *ptrFiredTable;
    assert(firedTable.currTime > (ptrNrnThread->_t - 0.1*ptrNrnThread->_dt));
    assert(firedTable.currTime < (ptrNrnThread->_t + 0.1*ptrNrnThread->_dt));
    assert(!firedTable.locked);
    for (int i = 0; i < size; ++i) {
        const NRNMPI_Spike& spk = spikes[i];
        const int stepDiff = (spk.spiketime - firedTable.currTime) / firedTable.dt;
        if (gid2in.find(spk.gid) == gid2in.end()) continue;
        if (stepDiff >= 0 && stepDiff > firedTable.maxAheadStep) {
            printf("warning: overahead spike occured, skipped\n");
            continue;
        } 
        else if (stepDiff < 0 && abs(stepDiff) > firedTable.maxDelayStep) {
            printf("warning: outdated spike occured, skipped\n");
            continue;
        }
        assert(recvSpikeSize < maxSpikeSize);
        cRecvSpikeGids[recvSpikeSize] = spk.gid;
        cRecvSpikeTimes[recvSpikeSize] = spk.spiketime;
        ++recvSpikeSize;
    }
    commitCpuRecvSpikesToFiredTable(cRecvSpikeGids, cRecvSpikeTimes, &recvSpikeSize);
}

void fireNetCons(NrnThread* ptrNrnThread, const double lowerBoundFactor, const double upperBoundFactor) {
    Instrumentor::phase p("fireNetCons");
    int firedNidSize = 0;
    {
    Instrumentor::phase p("TransferAndCollectFiredNidsFromFiredTable");
    {
    Instrumentor::phase p("memTransferBeforeCollectFiredNids");
    gpuErrchk(cudaMemset(
        (void*)(gPtrFiredNidSize),
        0,
        sizeof(int)
    ));
    }
    CollectFiredNidsFromFiredTable(gFiredNids, gFiredNidTimes, gPtrFiredNidSize, lowerBoundFactor, upperBoundFactor);
    {
    Instrumentor::phase p("memTransferAfterCollectFiredNids");
    gpuErrchk(cudaMemcpy(
        (void*)(&firedNidSize), 
        (void*)(gPtrFiredNidSize), 
        sizeof(int),
        cudaMemcpyDeviceToHost
    ));
    assert(firedNidSize <= maxFiredNidSize);
    if (firedNidSize > 0) {
        gpuErrchk(cudaMemcpy(
            (void*)(cFiredNids), 
            (void*)(gFiredNids), 
            firedNidSize*sizeof(cFiredNids[0]),
            cudaMemcpyDeviceToHost
        ));
        gpuErrchk(cudaMemcpy(
            (void*)(cFiredNidTimes), 
            (void*)(gFiredNidTimes), 
            firedNidSize*sizeof(cFiredNidTimes[0]),
            cudaMemcpyDeviceToHost
        ));
    }
    }
    }

    {
    Instrumentor::phase p("NetConsGetPntReceive");
    TRACE("firedNidSize = %d", firedNidSize);
    const double originalTime = ptrNrnThread->_t;
#ifdef CUDA_DELIVER_DEBUG
    std::sort(&cFiredNids[0], &cFiredNids[firedNidSize]);
    int redundantCnt = 0;
    for (int i = 1; i < firedNidSize; ++i) {
        if (cFiredNids[i] == cFiredNids[i-1]) ++redundantCnt;
    }
    TRACE("redundantCnt = %d", redundantCnt);
#endif
    for (int i = 0; i < firedNidSize; ++i) {
        NetCon* ptrNetCon = netcon_in_presyn_order_[cFiredNids[i]];
        ptrNetCon->deliver(cFiredNidTimes[i], nullptr, ptrNrnThread);
    }
    ptrNrnThread->_t = originalTime;
    }
}

void recordSpikes(const int* preSynIds, const int size, const NrnThread* ptrNrnThread) {
    const double time = ptrNrnThread->_t;
    for (int i = 0; i < size; ++i) {
        PreSyn* ptrPreSyn = ptrNrnThread->presyns + preSynIds[i];
        ptrPreSyn->record(time);
    }
}

void cpuMultiSend(const int* cFiredPreSynIds, const int size, const NrnThread* ptrNrnThread) {
    Instrumentor::phase p("cpuMultiSend");
    for (int i = 0; i < size; ++i) {
        const int preSynId = cFiredPreSynIds[i];
        const int multisendIdx = ptrSimpPreSyns->cMultisendIdxs[preSynId];
        if (multisendIdx < 0) continue;
        int* ranks = targets_phase1_ + multisendIdx;
        const int cntPhase1 = ranks[1];
        ranks += 2;
        NRNMPI_Spike spk;
        spk.gid = ptrSimpPreSyns->cOutputIdxs[preSynId];
        // for now, set the spike time as the nrnthread time, plus small teps
        // just as what NetCvod::check_thresh does
        // to be verified if this is correct
        spk.spiketime = ptrNrnThread->_t + teps;
        assert(spk.gid >= 0);
        TRACE("send spike (%d,%lf)", spk.gid, spk.spiketime);
        nrnmpi_multisend(&spk, cntPhase1, ranks, &multisendComm);
        multiSendCnt += cntPhase1;
    }
}

void cpuMultiRecv(int* cRecvSpikeGidBuffer, double* cRecvSpikeTimeBuffer, int* ptrSize, const int capacity) {
    Instrumentor::phase p = Instrumentor::phase("cpuMultiRecv");
    int& size = *ptrSize;
    const FiredTable& firedTable = *ptrFiredTable;
    NRNMPI_Spike spk;
    assert(!firedTable.locked);
    while(nrnmpi_multisend_single_advance(&spk, &multisendComm)) {
        TRACE("receive spike (%d,%lf)", spk.gid, spk.spiketime);
        ++multiRecvCnt;
        const int stepDiff = (spk.spiketime - firedTable.currTime) / firedTable.dt;
        if (stepDiff >= 0 && stepDiff > firedTable.maxAheadStep) {
            printf("warning: overahead spike occured, skipped\n");
            continue;
        } 
        else if (stepDiff < 0 && abs(stepDiff) > firedTable.maxDelayStep) {
            printf("warning: outdated spike occured, skipped\n");
            continue;
        }
        assert(size < capacity);
        cRecvSpikeGidBuffer[size] = spk.gid;
        cRecvSpikeTimeBuffer[size] = spk.spiketime;
        ++size;
    }
}

void cpuOutputEvent(const int* cFiredPreSynIds, const int size, const NrnThread* ptrNrnThread) {
    Instrumentor::phase p("cpuOutputEvent");
    const SimplifiedPreSyns& simpPreSyns = *ptrSimpPreSyns;
    for (int i = 0; i < size; ++i) {
        const int preSynId = cFiredPreSynIds[i];
        if (simpPreSyns.cOutputIdxs[preSynId] >= 0) {
            // for now, set the spike time as the nrnthread time, plus small teps
            // just as what NetCvod::check_thresh does
            // to be verified if this is correct
            double tt = ptrNrnThread->_t + teps;
            if (nrn_use_localgid_) {
                nrn_outputevent(simpPreSyns.cLocalGids[preSynId], tt);
            }
            else {
                nrn2ncs_outputevent(simpPreSyns.cOutputIdxs[preSynId], tt);
            }
        }
    }
}

void commitCpuRecvSpikesToFiredTable(int* cRecvSpikeGidBuffer, double* cRecvSpikeTimeBuffer, int* ptrRecvSpikeSize) {
    Instrumentor::phase pp("commitCpuRecvSpikesToFiredTable");
    int& recvSpikeSize = *ptrRecvSpikeSize;
    if (recvSpikeSize > 0) {
        {
        Instrumentor::phase p("transferBeforeCommitGidsWithTimes");
        gpuErrchk(cudaMemcpy(
            (void*)gRecvSpikeGids,
            (void*)cRecvSpikeGids,
            recvSpikeSize*sizeof(gRecvSpikeGids[0]),
            cudaMemcpyHostToDevice
        ));
        gpuErrchk(cudaMemcpy(
            (void*)gRecvSpikeTimes,
            (void*)cRecvSpikeTimes,
            recvSpikeSize*sizeof(gRecvSpikeTimes[0]),
            cudaMemcpyHostToDevice
        ));
        }
        {
        Instrumentor::phase p("CommitGidsWithTimesToFiredTable");
        CommitGidsWithTimesToFiredTable(gRecvSpikeGids, gRecvSpikeTimes, recvSpikeSize);
        }
    }
    recvSpikeSize = 0;
}

}  // namespace coreneuron
