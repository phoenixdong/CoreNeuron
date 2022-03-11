#pragma once

/**
 * Naming Conventions:
 *  1. variable cXXXX means the data resides on CPU side, variable gXXXX means the data resides on GPU side but the pointer itself resides in the CPU side, variable dXXXX means the pointer itself resides in the GPU side (means the variable actually locates in GPU device memory)
 *  2. If there is no 'c' or 'g' or 'd' in the beginning of the variable, then the variable 
 *     resides on CPU side by default;
 *  3. All the variables are named in Camel-Case style.
 * 
 **/
namespace coreneuron {

extern void test_fetch_array(const int* acc_data, const int size);

struct SimplifiedPreSyns {
    int* cNcCnts;
    int* gNcCnts;
    int* cNcIdxs;
    int* gNcIdxs;
    int* cMultisendIdxs;
    int* gMultisendIdxs; // set but not used so far
    int* cOutputIdxs;
    int* gOutputIdxs;    // set but not used so far
    unsigned char* cLocalGids;
    unsigned char* gLocalGids;   // set but not used so far
    int* cGidToLid;
    int* gGidToLid;
    int maxGid;
    int size;

    SimplifiedPreSyns();
    virtual ~SimplifiedPreSyns();
};

struct SimplifiedNetCons {
    double* cDelays;
    double* gDelays;
    int* cNids;
    int* gNids;
    int size;

    SimplifiedNetCons();
    virtual ~SimplifiedNetCons();
};

struct FiredTable {
    bool locked;
    double currTime;
    double dt;
    int maxStep;
    int maxDelayStep;
    int maxAheadStep;
    int currStep;
    int lineCapacity;
    int* gLineSizes;
    int* gFreshLineSizes;
    int* gData;

    FiredTable();
    virtual ~FiredTable();
};

extern SimplifiedPreSyns* ptrSimpPreSyns;
extern SimplifiedNetCons* ptrSimpNetCons;
extern FiredTable* ptrFiredTable;

void IncFiredTableTimeStep();
void IncFiredTableHalfTimeStep();
void CommitLidsToFiredTable(const int* gLids, const int size);
void CommitGidsWithTimesToFiredTable(const int* gGids, const double* gTimes, const int size);
void CollectFiredNidsFromFiredTable(int* gFiredNids, double* gFiredNidTimes, int* gPtrFiredNidSize, 
                                    const double lowerBoundFactor, const double upperBoundFactor);

}  // namespace coreneuron
