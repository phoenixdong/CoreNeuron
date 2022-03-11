#include "firedtable.h"
#include "util.h"

#include "coreneuron/utils/profile/profiler_interface.h"

#include <stdio.h>
namespace coreneuron {

__global__ 
void echo(
    const int* data,
    const int size
) {
    if ((threadIdx.x == 0) && (blockIdx.x == 0)) {
        printf("data size %d, first data %d\n", size, data[0]);
    }
}

void test_fetch_array(const int* acc_data, const int size) {
    echo<<<1, 1, 1>>>(acc_data, size);
}

SimplifiedPreSyns* ptrSimpPreSyns = nullptr;
SimplifiedNetCons* ptrSimpNetCons = nullptr;
FiredTable* ptrFiredTable = nullptr;

struct DeviceFiredTableInfo {
    double dDt;
    int dMaxStep;
    int dLineCapacity;
    int* dLineSizes;
    int* dData;
};
__device__ DeviceFiredTableInfo dFiredTableInfo;

SimplifiedPreSyns::SimplifiedPreSyns() :
    cNcCnts(nullptr),
    gNcCnts(nullptr),
    cNcIdxs(nullptr),
    gNcIdxs(nullptr),
    cMultisendIdxs(nullptr),
    gMultisendIdxs(nullptr),
    cOutputIdxs(nullptr),
    gOutputIdxs(nullptr),
    cLocalGids(nullptr),
    gLocalGids(nullptr),
    cGidToLid(nullptr),
    gGidToLid(nullptr),
    maxGid(0),
    size(0) 
{
}

SimplifiedPreSyns::~SimplifiedPreSyns()
{
    if (cNcCnts != nullptr) delete [] cNcCnts;
    if (cNcIdxs != nullptr) delete [] cNcIdxs;
    if (cMultisendIdxs != nullptr) delete [] cMultisendIdxs;
    if (cOutputIdxs != nullptr) delete [] cOutputIdxs;
    if (cLocalGids != nullptr) delete [] cLocalGids;
    if (cGidToLid != nullptr) delete [] cGidToLid;
    if (gNcCnts != nullptr) gpuErrchk(cudaFree(gNcCnts));
    if (gNcIdxs != nullptr) gpuErrchk(cudaFree(gNcIdxs));
    if (gMultisendIdxs != nullptr) gpuErrchk(cudaFree(gMultisendIdxs));
    if (gOutputIdxs != nullptr) gpuErrchk(cudaFree(gOutputIdxs));
    if (gLocalGids != nullptr) gpuErrchk(cudaFree(gLocalGids));
    if (gGidToLid != nullptr) gpuErrchk(cudaFree(gGidToLid));
}

SimplifiedNetCons::SimplifiedNetCons() :
    cDelays(nullptr),
    gDelays(nullptr),
    cNids(nullptr),
    gNids(nullptr),
    size(0)
{
}

SimplifiedNetCons::~SimplifiedNetCons()
{
    if (cDelays != nullptr) delete [] cDelays;
    if (cNids != nullptr) delete [] cNids;
    if (gDelays != nullptr) gpuErrchk(cudaFree(gDelays));
    if (gNids != nullptr) gpuErrchk(cudaFree(gNids));
}

FiredTable::FiredTable() :
    locked(false),
    currTime(0.),
    dt(0.),
    maxStep(0),
    maxDelayStep(0),
    maxAheadStep(0),
    currStep(0),
    lineCapacity(0),
    gLineSizes(nullptr),
    gFreshLineSizes(nullptr),
    gData(nullptr)
{
}

FiredTable::~FiredTable()
{
    if (gLineSizes != nullptr) gpuErrchk(cudaFree(gLineSizes));
    if (gFreshLineSizes != nullptr) gpuErrchk(cudaFree(gFreshLineSizes));
    if (gData != nullptr) gpuErrchk(cudaFree(gData));
}

static const int maxBlockSize = 128;

/* the gridDim and blockDim should all be 1, 
 * launched as <<<1, 1>>> 
 */
__global__
void kernelUpdateValue(
    const int value,            /* in */
    int* p                      /* out */
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *p = value;
    }
    __syncthreads();
}

/* the gridDim should be 1, and the blockDim should only have .x dimension, 
 * launched as <<<1, blkSize>>> 
 */
__global__
void kernelCommitLids(
    const int* src,             /* in */
    const int srcSize,          /* in */
    int* destBegin,             /* out */
    int* ptrDestSize            /* out */
) {
    int* dest = destBegin + *ptrDestSize;
    for (int i = threadIdx.x; i < srcSize; i += blockDim.x) {
        dest[i] = src[i];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        *ptrDestSize += srcSize;
    }
    __syncthreads();
}

/* the gridDim should be 1, and the blockDim should only have .x dimension, 
 * launched as <<<1, blkSize>>> 
 */
__global__
void kernelCommitGidsWithTimes(
    const int* srcGids,             /* in */
    const double* srcTimes,         /* in */
    const int srcSize,              /* in */
    const int* gidToLid,            /* in */
    const int lineCapacity,         /* in */
    const double currTime,          /* in */
    const double dt,                /* in */
    const int currStep,             /* in */
    const int maxStep,              /* in */
    const int maxDelayStep,         /* in */
    const int maxAheadStep,         /* in */
    int* destBegin,                 /* out */
    int* destSizes,                 /* out */
    int* commitSizes                /* out */
) {
    for (int i = threadIdx.x; i < srcSize; i += blockDim.x) {
        const double time = srcTimes[i];
        const int lid = gidToLid[srcGids[i]];
        assert(lid >= 0);
        const int stepDiff = (time - currTime) / dt;
        // in asyncronized communication, we allow the step from ahead & delayed timesteps
        const bool flag = (stepDiff == 0)
                       || ((stepDiff > 0) && ((maxAheadStep - stepDiff) >= 0))
                       || ((stepDiff < 0) && ((maxDelayStep + stepDiff) >= 0));
        assert(flag);
        const int step = (currStep + stepDiff + maxStep) % maxStep;
        const int offset = atomicAdd(destSizes+step, 1);
        assert(offset < lineCapacity);
        destBegin[step*lineCapacity+offset] = lid;
        atomicAdd(commitSizes+step, 1);
    }
    __syncthreads();
}

/* the gridDim should be two dimentional:
 *  - the gridDim.y dimension is for firedLines in firedTable, 
 *  - the gridDim.x dimension is for firedLids in a single line of firedTable
 * the blockDim should only have .x dimension, for a lid's netcons pointed by ncIdx and ncCnt
 * launched as <<<(gridDim.x, gridDim.y), blkSize>>> 
 */
__global__
void kernelCollectFiredNids(
    const int* firedLineLids,       /* in */
    const int* firedLineSizes,      /* in */
    const int* firedFreshLineSizes, /* in */
    const int lineCapacity,         /* in */
    const double currTime,          /* in */
    const double lowerBoundFactor,  /* in */
    const double upperBoundFactor,  /* in */
    const double dt,                /* in */
    const int currStep,             /* in */
    const int maxStep,              /* in */
    const int maxDelayStep,         /* in */
    const int maxAheadStep,         /* in */
    const int* lidNcIdxs,           /* in */
    const int* lidNcCnts,           /* in */
    const double* netconDelays,     /* in */
    const int* netconNids,          /* in */
    int* firedNids,                 /* out */
    double* firedNidTimes,          /* out */
    int* ptrFiredNidSize            /* out */
) {
    const int blockFiredCapacity = 500;
    __shared__ int blockFiredNids[blockFiredCapacity];
    __shared__ double blockFiredNidTimes[blockFiredCapacity];
    __shared__ volatile unsigned int blockNcIdx;
    __shared__ volatile unsigned int blockNcCnt;
    __shared__ volatile unsigned int blockFiredCnt;
    __shared__ volatile unsigned int blockFiredNidOffset;
    const double teps = 1e-10;

    for (int lineId = blockIdx.y; lineId < maxStep; lineId += gridDim.y) {
        const int lineSize = firedLineSizes[lineId];
        if (lineSize <= 0) continue;
        assert(lineSize <= lineCapacity);
        const int freshLineSize = firedFreshLineSizes[lineId];
        assert(freshLineSize <= lineSize);
        const int freshLineOffset = lineSize - freshLineSize;
        // const int maxAheadLineId = (currStep + maxAheadStep) % maxStep;
        // const double maxAheadLineTime = currTime + maxAheadStep * dt;
        // const double lineTime = maxAheadLineTime - ((maxAheadLineId - lineId + maxStep) % maxStep) * dt;
        // if (lineTime > currTime) continue;
        const int stepDiff = (currStep - lineId + maxStep) % maxStep;
        if (stepDiff > maxDelayStep) continue;
        const double lineTime = currTime - stepDiff * dt + teps;
        // const double lineTime = currTime - ((currStep - lineId + maxStep) % maxStep) * dt;
        for (int lineOffset = blockIdx.x; lineOffset < lineSize; lineOffset += gridDim.x) {
            if (threadIdx.x == 0) {
                const int lid = firedLineLids[lineId*lineCapacity+lineOffset];
                blockNcIdx = lidNcIdxs[lid];
                blockNcCnt = lidNcCnts[lid];
                blockFiredCnt = 0;
                blockFiredNidOffset = 0;
            }
            __syncthreads();
            for (int ncOffset = threadIdx.x; ncOffset < blockNcCnt; ncOffset += blockDim.x) {
                const double ncDelay = netconDelays[ncOffset+blockNcIdx];
                const double fireTime = ncDelay + lineTime;
                // bool flag = (fireTime <= (currTime + 0.5*dt))
                //         //  && ((fireTime > (currTime - 0.5*dt)) || (lineOffset >= freshLineOffset));
                //          && ((fireTime > currTime) || (lineOffset >= freshLineOffset));
                bool flag = (fireTime <= (currTime + upperBoundFactor*dt))
                         && ((fireTime > currTime + lowerBoundFactor*dt) || (lineOffset >= freshLineOffset));
                if (flag) {
                    const int ncNid = netconNids[ncOffset+blockNcIdx];
#if __CUDA_ARCH__ >= 600
                    const int firedOffset = atomicAdd_block((int*)&blockFiredCnt, 1);
#else
                    const int firedOffset = atomicAdd((int*)&blockFiredCnt, 1);
#endif
                    assert(firedOffset < blockFiredCapacity);
                    blockFiredNids[firedOffset] = ncNid;
                    blockFiredNidTimes[firedOffset] = fireTime;
                }
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                blockFiredNidOffset = atomicAdd(ptrFiredNidSize, blockFiredCnt);
            }
            __syncthreads();
            for (int i = threadIdx.x; i < blockFiredCnt; i += blockDim.x) {
                const int firedNidOffset = i + blockFiredNidOffset;
                firedNids[firedNidOffset] = blockFiredNids[i];
                firedNidTimes[firedNidOffset] = blockFiredNidTimes[i];
            }
            __syncthreads();
        }
    }
}

void IncFiredTableTimeStep() {
    Instrumentor::phase p("IncFiredTableTimeStep");
    FiredTable& firedTable = *ptrFiredTable;
    firedTable.currTime += firedTable.dt;
    firedTable.currStep = roundedInc(firedTable.currStep, firedTable.maxStep);
#ifdef SYNCRONIZED_COMMUNICATION
    assert(false);
#else
    kernelUpdateValue<<<1, 1>>>(
        0,
        &firedTable.gLineSizes[(firedTable.currStep+firedTable.maxAheadStep)%firedTable.maxStep]
    );
#endif
    gpuErrchk(cudaPeekAtLastError());
}

void IncFiredTableHalfTimeStep() {
    static int halfTimeStepCnt = 0;
    Instrumentor::phase p("IncFiredTableHalfTimeStep");
    ++halfTimeStepCnt;
    FiredTable& firedTable = *ptrFiredTable;
    if ((halfTimeStepCnt & 1) != 0) {
        firedTable.currTime += firedTable.dt * 0.5;
        firedTable.locked = true;
    }
    else {
        firedTable.currTime += firedTable.dt * 0.5;
        firedTable.currStep = roundedInc(firedTable.currStep, firedTable.maxStep);
#ifdef SYNCRONIZED_COMMUNICATION
        assert(false);
#else
        kernelUpdateValue<<<1, 1>>>(
            0,
            &firedTable.gLineSizes[(firedTable.currStep+firedTable.maxAheadStep)%firedTable.maxStep]
        );
#endif
        gpuErrchk(cudaPeekAtLastError());
        firedTable.locked = false;
    }
}

void CommitLidsToFiredTable(const int* gLids, const int size) {
    Instrumentor::phase p("CommitLidsToFiredTable");
    FiredTable& firedTable = *ptrFiredTable;
    assert(!firedTable.locked);
    assert(size > 0);
    assert(size <= firedTable.lineCapacity);
    int blockSize = (size < maxBlockSize) ? size : maxBlockSize;
    kernelCommitLids<<<1, blockSize>>>(
        gLids,
        size,
        &firedTable.gData[firedTable.currStep*firedTable.lineCapacity],
        &firedTable.gLineSizes[firedTable.currStep]
    );
    gpuErrchk(cudaPeekAtLastError());
}

void CommitGidsWithTimesToFiredTable(const int* gGids, const double* gTimes, const int size) {
    SimplifiedPreSyns& simpPreSyns = *ptrSimpPreSyns;
    FiredTable& firedTable = *ptrFiredTable;
    assert(!firedTable.locked);
    assert(size > 0);
    assert(size <= firedTable.lineCapacity);
    int blockSize = (size < maxBlockSize) ? size : maxBlockSize;
#ifdef SYNCRONIZED_COMMUNICATION
    assert(false);
#else
    kernelCommitGidsWithTimes<<<1, blockSize>>>(
        gGids,                          // const int* srcGids,             /* in */
        gTimes,                         // const double* srcTimes,         /* in */
        size,                           // const int srcSize,              /* in */
        simpPreSyns.gGidToLid,          // const int* gidToLid,            /* in */
        firedTable.lineCapacity,        // const int lineCapacity,         /* in */
        firedTable.currTime,            // const double currTime,          /* in */
        firedTable.dt,                  // const double dt,                /* in */
        firedTable.currStep,            // const int currStep,             /* in */
        firedTable.maxStep,             // const int maxStep,              /* in */
        firedTable.maxDelayStep,        // const int maxDelayStep,         /* in */
        firedTable.maxAheadStep,        // const int maxAheadStep,         /* in */
        firedTable.gData,               // int* destBegin,                 /* out */
        firedTable.gLineSizes,          // int* destSizes,                 /* out */
        firedTable.gFreshLineSizes      // int* commitSizes                /* out */
    );
#endif
    gpuErrchk(cudaPeekAtLastError());
}

void CollectFiredNidsFromFiredTable(
    int* gFiredNids,                    /* out */ 
    double* gFiredNidTimes,             /* out */
    int* gPtrFiredNidSize,              /* out */
    const double lowerBoundFactor,      /* in */
    const double upperBoundFactor       /* in */
) {
    Instrumentor::phase p("CollectFiredNidsFromFiredTable");
    SimplifiedPreSyns& simpPreSyns = *ptrSimpPreSyns;
    SimplifiedNetCons& simpNetCons = *ptrSimpNetCons;
    FiredTable& firedTable = *ptrFiredTable;
    assert(!firedTable.locked);
    assert(lowerBoundFactor <= upperBoundFactor);

    // const double currTime = firedTable.currTime;
    // firedTable.currTime = currTime + 0.5 * firedTable.dt;

    dim3 gridSize;
    gridSize.y = 128;
    gridSize.x = 128;
    int blockSize = maxBlockSize;
#ifdef SYNCRONIZED_COMMUNICATION
    assert(false);
#else
    kernelCollectFiredNids<<<gridSize, blockSize>>>(
        firedTable.gData,               // const int* firedLineLids,       /* in */
        firedTable.gLineSizes,          // const int* firedLineSizes,      /* in */
        firedTable.gFreshLineSizes,     // const int* firedFreshLineSizes, /* in */
        firedTable.lineCapacity,        // const int lineCapacity,         /* in */
        firedTable.currTime,            // const double currTime,          /* in */
        lowerBoundFactor,               // const double lowerBoundFactor,  /* in */
        upperBoundFactor,               // const double upperBoundFactor,  /* in */
        firedTable.dt,                  // const double dt,                /* in */
        firedTable.currStep,            // const int currStep,             /* in */
        firedTable.maxStep,             // const int maxStep,              /* in */
        firedTable.maxDelayStep,        // const int maxDelayStep,         /* in */
        firedTable.maxAheadStep,        // const int maxAheadStep,         /* in */
        simpPreSyns.gNcIdxs,            // const int* lidNcIdxs,           /* in */
        simpPreSyns.gNcCnts,            // const int* lidNcCnts,           /* in */
        simpNetCons.gDelays,            // const double* netconDelays,     /* in */
        simpNetCons.gNids,              // const int* netconNids,          /* in */
        gFiredNids,                     // int* firedNids,                 /* out */
        gFiredNidTimes,                 // double* firedNidTimes,          /* out */
        gPtrFiredNidSize                // int* ptrFiredNidSize            /* out */
    );
#endif
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaMemset(
        (void*)(firedTable.gFreshLineSizes),
        0,
        firedTable.maxStep*sizeof(firedTable.gFreshLineSizes[0])
    ));
    // firedTable.currTime = currTime;
}

}  // namespace coreneuron
