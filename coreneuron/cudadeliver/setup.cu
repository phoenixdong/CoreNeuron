#include "setup.h"
#include "util.h"
#include "firedtable.h"
#include "cudadeliver.h"
#include "log.h"

#include "coreneuron/nrniv/nrniv_decl.h"
// #include "coreneuron/apps/corenrn_parameters.hpp"

#include <openacc.h>

/**
 * The implementation assumes:
 *  - openacc enabled
 *  - OMP_NUM_THREADS = 1, which means only nrn_threads[0] has valid data
**/

namespace coreneuron {

static bool isSetUp = false;
static double maxDelay = -1.;
static double timestep = -1.;
static double totalSizeInMb = 0.;
static double cudadeliverSizeInMb = 0.;

void CudaDeliverSetup(const NrnThread* ptrNrnThread);
void CudaDeliverCleanup();
void setupDevice();
void setupNetwork(const NrnThread* ptrNrnThread);
void setupFiredTable(const NrnThread* ptrNrnThread);
int calcMaxDelayStep();
int calcMaxAheadStep();
int calcFiredTableStep();
int calcMaxStepSpike(const NrnThread* ptrNrnThread);
void setupCommunication(const NrnThread* ptrNrnThread);
void memoryReport();

void CudaDeliverSetup(const NrnThread* ptrNrnThread) {
    assert(!isSetUp);
    isSetUp = true;
    timestep = ptrNrnThread->_dt;

    setupDevice();
    setupNetwork(ptrNrnThread);
    setupFiredTable(ptrNrnThread);
    setupCommunication(ptrNrnThread);

    memoryReport();
}

void CudaDeliverCleanup() {
    if (ptrSimpPreSyns != nullptr) delete ptrSimpPreSyns;
    if (ptrSimpNetCons != nullptr) delete ptrSimpNetCons;
    if (ptrFiredTable != nullptr) delete ptrFiredTable;
    if (cFiredPreSynIds != nullptr) delete [] cFiredPreSynIds;
    if (cRecvSpikeGids != nullptr) delete [] cRecvSpikeGids;
    if (cRecvSpikeTimes != nullptr) delete [] cRecvSpikeTimes;
    if (cFiredNids != nullptr) delete [] cFiredNids;
    if (cFiredNidTimes != nullptr) delete [] cFiredNidTimes;
    if (gRecvSpikeGids != nullptr) gpuErrchk(cudaFree(gRecvSpikeGids));
    if (gRecvSpikeTimes != nullptr) gpuErrchk(cudaFree(gRecvSpikeTimes));
    if (gPtrFiredNidSize != nullptr) gpuErrchk(cudaFree(gPtrFiredNidSize));
    if (gFiredNids != nullptr) gpuErrchk(cudaFree(gFiredNids));
    if (gFiredNidTimes != nullptr) gpuErrchk(cudaFree(gFiredNidTimes));
}

void setupDevice() {
    int deviceNum = acc_get_device_num(acc_device_nvidia);
    gpuErrchk(cudaSetDevice(deviceNum));
}

void setupNetwork(const NrnThread* ptrNrnThread) {
    ptrSimpPreSyns = new SimplifiedPreSyns;
    SimplifiedPreSyns& simpPreSyns = *ptrSimpPreSyns;
    ptrSimpNetCons = new SimplifiedNetCons;
    SimplifiedNetCons& simpNetCons = *ptrSimpNetCons;

    int maxInGid = gid2in.rbegin()->first;
    int maxOutGid = gid2out.rbegin()->first;
    int maxGid = (maxInGid > maxOutGid) ? maxInGid : maxOutGid;
    simpPreSyns.maxGid = maxGid;
    simpPreSyns.cGidToLid = new int[maxGid+1];
    memset((void*)(simpPreSyns.cGidToLid), -1, (maxGid+1)*sizeof(int));
    simpPreSyns.size = ptrNrnThread->ncell + gid2in.size();
    simpPreSyns.cNcCnts = new int[simpPreSyns.size];
    simpPreSyns.cNcIdxs = new int[simpPreSyns.size];
    simpPreSyns.cMultisendIdxs = new int[simpPreSyns.size];
    memset((void*)simpPreSyns.cMultisendIdxs, -1, simpPreSyns.size*sizeof(int));
    simpPreSyns.cOutputIdxs = new int[simpPreSyns.size];
    memset((void*)simpPreSyns.cOutputIdxs, -1, simpPreSyns.size*sizeof(int));
    simpPreSyns.cLocalGids = new unsigned char[simpPreSyns.size];
    memset((void*)simpPreSyns.cLocalGids, -1, simpPreSyns.size*sizeof(unsigned char));
    int lid = 0;
    int nid = 0;
    std::vector<double> cDelays;
    std::vector<int> cNids;
    for (int i = 0; i < ptrNrnThread->ncell; ++i) {
        const PreSyn* ptrPreSyn = ptrNrnThread->presyns + i;
        simpPreSyns.cNcIdxs[lid] = nid;
        for (int j = 0; j < ptrPreSyn->nc_cnt_; ++j) {
            const NetCon* ptrNetCon = netcon_in_presyn_order_[ptrPreSyn->nc_index_+j];
            if (ptrNetCon->active_ && ptrNetCon->target_) {
                cDelays.push_back(ptrNetCon->delay_);
                cNids.push_back(ptrPreSyn->nc_index_+j);
                ++nid;
            }
        }
        simpPreSyns.cNcCnts[lid] = nid - simpPreSyns.cNcIdxs[lid];
        simpPreSyns.cMultisendIdxs[lid] = ptrPreSyn->multisend_index_;
        simpPreSyns.cOutputIdxs[lid] = ptrPreSyn->output_index_;
        simpPreSyns.cLocalGids[lid] = ptrPreSyn->localgid_;
        if (ptrPreSyn->gid_ < 0) {
            ++lid;
            continue;
        }
        simpPreSyns.cGidToLid[ptrPreSyn->gid_] = lid;
        ++lid;
    }
    for (const auto pair : gid2in) {
        assert(simpPreSyns.cGidToLid[pair.first] < 0);
        simpPreSyns.cGidToLid[pair.first] = lid;
        const InputPreSyn* ptrInputPreSyn = pair.second;
        simpPreSyns.cNcIdxs[lid] = nid;
        for (int j = 0; j < ptrInputPreSyn->nc_cnt_; ++j) {
            const NetCon* ptrNetCon = netcon_in_presyn_order_[ptrInputPreSyn->nc_index_+j];
            if (ptrNetCon->active_ && ptrNetCon->target_) {
                cDelays.push_back(ptrNetCon->delay_);
                cNids.push_back(ptrInputPreSyn->nc_index_+j);
                ++nid;
            }
        }
        simpPreSyns.cNcCnts[lid] = nid - simpPreSyns.cNcIdxs[lid];
        ++lid;
    }
    assert(lid == simpPreSyns.size);
#ifdef CUDA_DELIVER_DEBUG
    int maxNcCnts = 0;
    for (int i = 0; i < simpPreSyns.size; ++i) {
        maxNcCnts = simpPreSyns.cNcCnts[i] > maxNcCnts ? simpPreSyns.cNcCnts[i] : maxNcCnts;
    }
    TRACE("maxNcCnts = %d", maxNcCnts);
#endif
    simpNetCons.size = cDelays.size();
    assert(nid == simpNetCons.size);
    simpNetCons.cDelays = new double[simpNetCons.size];
    simpNetCons.cNids = new int[simpNetCons.size];
    for (int i = 0; i < simpNetCons.size; ++i) {
        simpNetCons.cDelays[i] = cDelays[i];
        simpNetCons.cNids[i] = cNids[i];
        maxDelay = (maxDelay >= cDelays[i]) ? maxDelay : cDelays[i];
    }

    gpuErrchk(cudaMalloc(
        (void**)(&(simpPreSyns.gNcCnts)), 
        simpPreSyns.size*sizeof(simpPreSyns.gNcCnts[0])
    ));
    totalSizeInMb += sizeInMb(simpPreSyns.size*sizeof(simpPreSyns.gNcCnts[0]));
    cudadeliverSizeInMb += sizeInMb(simpPreSyns.size*sizeof(simpPreSyns.gNcCnts[0]));
    INFO("simpPreSyns.gNcCnts = %.3lfMB, totalSize = %.3lfMB", 
        sizeInMb(simpPreSyns.size*sizeof(simpPreSyns.gNcCnts[0])),
        totalSizeInMb
    );
    gpuErrchk(cudaMemcpy(
        (void*)(simpPreSyns.gNcCnts), 
        (void*)(simpPreSyns.cNcCnts), 
        simpPreSyns.size*sizeof(simpPreSyns.gNcCnts[0]),
        cudaMemcpyHostToDevice
    ));
    gpuErrchk(cudaMalloc(
        (void**)(&(simpPreSyns.gNcIdxs)), 
        simpPreSyns.size*sizeof(simpPreSyns.gNcIdxs[0])
    ));
    totalSizeInMb += sizeInMb(simpPreSyns.size*sizeof(simpPreSyns.gNcIdxs[0]));
    cudadeliverSizeInMb += sizeInMb(simpPreSyns.size*sizeof(simpPreSyns.gNcIdxs[0]));
    INFO("simpPreSyns.gNcIdxs = %.3lfMB, totalSize = %.3lfMB", 
        sizeInMb(simpPreSyns.size*sizeof(simpPreSyns.gNcIdxs[0])),
        totalSizeInMb
    );
    gpuErrchk(cudaMemcpy(
        (void*)(simpPreSyns.gNcIdxs), 
        (void*)(simpPreSyns.cNcIdxs), 
        simpPreSyns.size*sizeof(simpPreSyns.gNcIdxs[0]),
        cudaMemcpyHostToDevice
    ));
#ifdef GPU_SEND
    gpuErrchk(cudaMalloc(
        (void**)(&(simpPreSyns.gMultisendIdxs)), 
        simpPreSyns.size*sizeof(simpPreSyns.gMultisendIdxs[0])
    ));
    totalSizeInMb += sizeInMb(simpPreSyns.size*sizeof(simpPreSyns.gMultisendIdxs[0]));
    cudadeliverSizeInMb += sizeInMb(simpPreSyns.size*sizeof(simpPreSyns.gMultisendIdxs[0]));
    INFO("simpPreSyns.gMultisendIdxs = %.3lfMB, totalSize = %.3lfMB", 
        sizeInMb(simpPreSyns.size*sizeof(simpPreSyns.gMultisendIdxs[0])),
        totalSizeInMb
    );
    gpuErrchk(cudaMemcpy(
        (void*)(simpPreSyns.gMultisendIdxs), 
        (void*)(simpPreSyns.cMultisendIdxs), 
        simpPreSyns.size*sizeof(simpPreSyns.gMultisendIdxs[0]),
        cudaMemcpyHostToDevice
    ));
    gpuErrchk(cudaMalloc(
        (void**)(&(simpPreSyns.gOutputIdxs)), 
        simpPreSyns.size*sizeof(simpPreSyns.gOutputIdxs[0])
    ));
    totalSizeInMb += sizeInMb(simpPreSyns.size*sizeof(simpPreSyns.gOutputIdxs[0]));
    cudadeliverSizeInMb += sizeInMb(simpPreSyns.size*sizeof(simpPreSyns.gOutputIdxs[0]));
    INFO("simpPreSyns.gOutputIdxs = %.3lfMB, totalSize = %.3lfMB", 
        sizeInMb(simpPreSyns.size*sizeof(simpPreSyns.gOutputIdxs[0])),
        totalSizeInMb
    );
    gpuErrchk(cudaMemcpy(
        (void*)(simpPreSyns.gOutputIdxs), 
        (void*)(simpPreSyns.cOutputIdxs), 
        simpPreSyns.size*sizeof(simpPreSyns.gOutputIdxs[0]),
        cudaMemcpyHostToDevice
    ));
    gpuErrchk(cudaMalloc(
        (void**)(&(simpPreSyns.gLocalGids)), 
        simpPreSyns.size*sizeof(simpPreSyns.gLocalGids[0])
    ));
    totalSizeInMb += sizeInMb(simpPreSyns.size*sizeof(simpPreSyns.gLocalGids[0]));
    cudadeliverSizeInMb += sizeInMb(simpPreSyns.size*sizeof(simpPreSyns.gLocalGids[0]));
    INFO("simpPreSyns.gLocalGids = %.3lfMB, totalSize = %.3lfMB", 
        sizeInMb(simpPreSyns.size*sizeof(simpPreSyns.gLocalGids[0])),
        totalSizeInMb
    );
    gpuErrchk(cudaMemcpy(
        (void*)(simpPreSyns.gLocalGids), 
        (void*)(simpPreSyns.cLocalGids), 
        simpPreSyns.size*sizeof(simpPreSyns.gLocalGids[0]),
        cudaMemcpyHostToDevice
    ));
#endif
    gpuErrchk(cudaMalloc(
        (void**)(&(simpPreSyns.gGidToLid)), 
        (simpPreSyns.maxGid+1)*sizeof(simpPreSyns.gGidToLid[0])
    ));
    totalSizeInMb += sizeInMb((simpPreSyns.maxGid+1)*sizeof(simpPreSyns.gGidToLid[0]));
    cudadeliverSizeInMb += sizeInMb((simpPreSyns.maxGid+1)*sizeof(simpPreSyns.gGidToLid[0]));
    INFO("simpPreSyns.gGidToLid = %.3lfMB, totalSize = %.3lfMB", 
        sizeInMb((simpPreSyns.maxGid+1)*sizeof(simpPreSyns.gGidToLid[0])),
        totalSizeInMb
    );
    gpuErrchk(cudaMemcpy(
        (void*)(simpPreSyns.gGidToLid), 
        (void*)(simpPreSyns.cGidToLid), 
        (simpPreSyns.maxGid+1)*sizeof(simpPreSyns.gGidToLid[0]),
        cudaMemcpyHostToDevice
    ));

    gpuErrchk(cudaMalloc(
        (void**)(&(simpNetCons.gDelays)), 
        simpNetCons.size*sizeof(simpNetCons.gDelays[0])
    ));
    totalSizeInMb += sizeInMb(simpNetCons.size*sizeof(simpNetCons.gDelays[0]));
    cudadeliverSizeInMb += sizeInMb(simpNetCons.size*sizeof(simpNetCons.gDelays[0]));
    INFO("simpNetCons.gDelays = %.3lfMB, totalSize = %.3lfMB", 
        sizeInMb(simpNetCons.size*sizeof(simpNetCons.gDelays[0])),
        totalSizeInMb
    );
    gpuErrchk(cudaMemcpy(
        (void*)(simpNetCons.gDelays), 
        (void*)(simpNetCons.cDelays), 
        simpNetCons.size*sizeof(simpNetCons.gDelays[0]),
        cudaMemcpyHostToDevice
    ));
    gpuErrchk(cudaMalloc(
        (void**)(&(simpNetCons.gNids)), 
        simpNetCons.size*sizeof(simpNetCons.gNids[0])
    ));
    totalSizeInMb += sizeInMb(simpNetCons.size*sizeof(simpNetCons.gNids[0]));
    cudadeliverSizeInMb += sizeInMb(simpNetCons.size*sizeof(simpNetCons.gNids[0]));
    INFO("simpNetCons.gNids = %.3lfMB, totalSize = %.3lfMB", 
        sizeInMb(simpNetCons.size*sizeof(simpNetCons.gNids[0])),
        totalSizeInMb
    );
    gpuErrchk(cudaMemcpy(
        (void*)(simpNetCons.gNids), 
        (void*)(simpNetCons.cNids), 
        simpNetCons.size*sizeof(simpNetCons.gNids[0]),
        cudaMemcpyHostToDevice
    ));
}

void setupFiredTable(const NrnThread* ptrNrnThread) {
    ptrFiredTable = new FiredTable;
    FiredTable& firedTable = *ptrFiredTable;

    firedTable.dt = ptrNrnThread->_dt;
    firedTable.maxStep = calcFiredTableStep();
    firedTable.maxDelayStep = calcMaxDelayStep();
    firedTable.maxAheadStep = calcMaxAheadStep();
    firedTable.currStep = 0;
    firedTable.lineCapacity = alignUp(calcMaxStepSpike(ptrNrnThread), 32);
    INFO("firedTable.maxStep = %d, maxDelayStep = %d, maxAheadStep = %d, lineCapacity = %d",
        firedTable.maxStep,
        firedTable.maxDelayStep,
        firedTable.maxAheadStep,
        firedTable.lineCapacity
    );
    gpuErrchk(cudaMalloc(
        (void**)(&(firedTable.gLineSizes)),
        firedTable.maxStep*sizeof(firedTable.gLineSizes[0])
    ));
    totalSizeInMb += sizeInMb(firedTable.maxStep*sizeof(firedTable.gLineSizes[0]));
    cudadeliverSizeInMb += sizeInMb(firedTable.maxStep*sizeof(firedTable.gLineSizes[0]));
    INFO("firedTable.gLineSizes = %.3lfMB, totalSize = %.3lfMB", 
        sizeInMb(firedTable.maxStep*sizeof(firedTable.gLineSizes[0])),
        totalSizeInMb
    );
    gpuErrchk(cudaMemset(
        (void*)(firedTable.gLineSizes),
        0,
        firedTable.maxStep*sizeof(firedTable.gLineSizes[0])
    ));
    gpuErrchk(cudaMalloc(
        (void**)(&(firedTable.gFreshLineSizes)),
        firedTable.maxStep*sizeof(firedTable.gFreshLineSizes[0])
    ));
    totalSizeInMb += sizeInMb(firedTable.maxStep*sizeof(firedTable.gFreshLineSizes[0]));
    cudadeliverSizeInMb += sizeInMb(firedTable.maxStep*sizeof(firedTable.gFreshLineSizes[0]));
    INFO("firedTable.gFreshLineSizes = %.3lfMB, totalSize = %.3lfMB", 
        sizeInMb(firedTable.maxStep*sizeof(firedTable.gFreshLineSizes[0])),
        totalSizeInMb
    );
    gpuErrchk(cudaMemset(
        (void*)(firedTable.gFreshLineSizes),
        0,
        firedTable.maxStep*sizeof(firedTable.gFreshLineSizes[0])
    ));
    gpuErrchk(cudaMalloc(
        (void**)(&(firedTable.gData)),
        firedTable.maxStep*firedTable.lineCapacity*sizeof(firedTable.gData[0])
    ));
    totalSizeInMb += sizeInMb(firedTable.maxStep*firedTable.lineCapacity*sizeof(firedTable.gData[0]));
    cudadeliverSizeInMb += sizeInMb(firedTable.maxStep*firedTable.lineCapacity*sizeof(firedTable.gData[0]));
    INFO("firedTable.gData = %.3lfMB, totalSize = %.3lfMB", 
        sizeInMb(firedTable.maxStep*firedTable.lineCapacity*sizeof(firedTable.gData[0])),
        totalSizeInMb
    );
    gpuErrchk(cudaMemset(
        (void*)(firedTable.gData),
        0,
        firedTable.maxStep*firedTable.lineCapacity*sizeof(firedTable.gData[0])
    ));
}

/* TODO: calculate according to SimplifiedNetCons.cDelays */
int calcMaxDelayStep() {
    assert(maxDelay >= 0);
    assert(timestep > 0);
    return (int)(maxDelay / timestep + 1);
}

int calcMaxAheadStep() {
    return 32;
}

int calcFiredTableStep() {
    return calcMaxDelayStep() + calcMaxAheadStep() + 1;
}

int calcMaxStepSpike(const NrnThread* ptrNrnThread) {
    return ptrNrnThread->ncell + gid2in.size();
}

int calcMaxFiredNetConSize(const NrnThread* ptrNrnThread) {
    const int calc = calcFiredTableStep() * calcMaxStepSpike(ptrNrnThread) / 4;
    const int max = 256 * 1024 * 1024;
    return calc <= max ? calc : max;
}

void setupCommunication(const NrnThread* ptrNrnThread) {
    MPI_Comm_dup(MPI_COMM_WORLD, &multisendComm);
    multiSendCnt = 0;
    multiRecvCnt = 0;

    maxSpikeSize = calcMaxStepSpike(ptrNrnThread);
    INFO("cudaDeliver maxSpikeSize = %d", maxSpikeSize);
    cFiredPreSynIds = new int[maxSpikeSize];
    recvSpikeSize = 0;
    cRecvSpikeGids = new int[maxSpikeSize];
    cRecvSpikeTimes = new double[maxSpikeSize];
    gpuErrchk(cudaMalloc(
        (void**)(&gRecvSpikeGids),
        maxSpikeSize*sizeof(gRecvSpikeGids[0])
    ));
    totalSizeInMb += sizeInMb(maxSpikeSize*sizeof(gRecvSpikeGids[0]));
    cudadeliverSizeInMb += sizeInMb(maxSpikeSize*sizeof(gRecvSpikeGids[0]));
    INFO("gRecvSpikeGids = %.3lfMB, totalSize = %.3lfMB", 
        sizeInMb(maxSpikeSize*sizeof(gRecvSpikeGids[0])),
        totalSizeInMb
    );
    gpuErrchk(cudaMemset(
        (void*)(gRecvSpikeGids),
        0,
        maxSpikeSize*sizeof(gRecvSpikeGids[0])
    ));
    gpuErrchk(cudaMalloc(
        (void**)(&gRecvSpikeTimes),
        maxSpikeSize*sizeof(gRecvSpikeTimes[0])
    ));
    totalSizeInMb += sizeInMb(maxSpikeSize*sizeof(gRecvSpikeTimes[0]));
    cudadeliverSizeInMb += sizeInMb(maxSpikeSize*sizeof(gRecvSpikeTimes[0]));
    INFO("gRecvSpikeTimes = %.3lfMB, totalSize = %.3lfMB", 
        sizeInMb(maxSpikeSize*sizeof(gRecvSpikeTimes[0])),
        totalSizeInMb
    );
    gpuErrchk(cudaMemset(
        (void*)(gRecvSpikeTimes),
        0,
        maxSpikeSize*sizeof(gRecvSpikeTimes[0])
    ));

    maxFiredNidSize = alignUp(calcMaxFiredNetConSize(ptrNrnThread), 32);
    INFO("maxFiredNidSize = %d", maxFiredNidSize);
    gpuErrchk(cudaMalloc(
        (void**)(&gPtrFiredNidSize),
        32*sizeof(gPtrFiredNidSize[0])
    ));
    totalSizeInMb += sizeInMb(32*sizeof(gPtrFiredNidSize[0]));
    cudadeliverSizeInMb += sizeInMb(32*sizeof(gPtrFiredNidSize[0]));
    INFO("gPtrFiredNidSize = %.3lfMB, totalSize = %.3lfMB", 
        sizeInMb(32*sizeof(gPtrFiredNidSize[0])),
        totalSizeInMb
    );
    gpuErrchk(cudaMemset(
        (void*)(gPtrFiredNidSize),
        0,
        32*sizeof(gPtrFiredNidSize[0])
    ));
    cFiredNids = new int[maxFiredNidSize];
    cFiredNidTimes = new double[maxFiredNidSize];
    gpuErrchk(cudaMalloc(
        (void**)(&gFiredNids),
        maxFiredNidSize*sizeof(gFiredNids[0])
    ));
    totalSizeInMb += sizeInMb(maxFiredNidSize*sizeof(gFiredNids[0]));
    cudadeliverSizeInMb += sizeInMb(maxFiredNidSize*sizeof(gFiredNids[0]));
    INFO("gFiredNids = %.3lfMB, totalSize = %.3lfMB", 
        sizeInMb(maxFiredNidSize*sizeof(gFiredNids[0])),
        totalSizeInMb
    );
    gpuErrchk(cudaMemset(
        (void*)(gFiredNids),
        0,
        maxFiredNidSize*sizeof(gFiredNids[0])
    ));
    gpuErrchk(cudaMalloc(
        (void**)(&gFiredNidTimes),
        maxFiredNidSize*sizeof(gFiredNidTimes[0])
    ));
    totalSizeInMb += sizeInMb(maxFiredNidSize*sizeof(gFiredNidTimes[0]));
    cudadeliverSizeInMb += sizeInMb(maxFiredNidSize*sizeof(gFiredNidTimes[0]));
    INFO("gFiredNidTimes = %.3lfMB, totalSize = %.3lfMB", 
        sizeInMb(maxFiredNidSize*sizeof(gFiredNidTimes[0])),
        totalSizeInMb
    );
    gpuErrchk(cudaMemset(
        (void*)(gFiredNidTimes),
        0,
        maxFiredNidSize*sizeof(gFiredNidTimes[0])
    ));
}

void memoryReport() {
    size_t freeMemoryInByte, totalMemoryInByte;
    gpuErrchk(cudaMemGetInfo(&freeMemoryInByte, &totalMemoryInByte));
    size_t usedMemoryInByte = totalMemoryInByte - freeMemoryInByte;
    double usedMemoryInMb = sizeInMb(usedMemoryInByte);
    INFO("usedMemorySize = %.3lfMB, totalSize = %.3lfMB(%.2lf%% of usedMemory), cudadeliverSize = %.3lfMB(%.2lf%% of usedMemory)",
        usedMemoryInMb,
        totalSizeInMb, totalSizeInMb/usedMemoryInMb*100,
        cudadeliverSizeInMb, cudadeliverSizeInMb/usedMemoryInMb*100
    );
}

}  // namespace coreneuron
