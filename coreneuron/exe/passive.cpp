/*********************************************************
Model Name      : pas
Filename        : passive.mod
NMODL Version   : 6.2.0
Vectorized      : true
Threadsafe      : true
Created         : Tue Jan  4 11:50:16 2022
Backend         : C-OpenAcc (api-compatibility)
NMODL Compiler  : 0.3 [3e960d7d 2021-12-23 15:47:17 +0100]
*********************************************************/

#include <math.h>
#include "nmodl/fast_math.hpp" // extend math with some useful functions
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <coreneuron/utils/offload.hpp>
#include <cuda_runtime_api.h>

#include <coreneuron/mechanism/mech/cfile/scoplib.h>
#include <coreneuron/nrnconf.h>
#include <coreneuron/sim/multicore.hpp>
#include <coreneuron/mechanism/register_mech.hpp>
#include <coreneuron/gpu/nrn_acc_manager.hpp>
#include <coreneuron/utils/randoms/nrnran123.h>
#include <coreneuron/nrniv/nrniv_decl.h>
#include <coreneuron/utils/ivocvect.hpp>
#include <coreneuron/utils/nrnoc_aux.hpp>
#include <coreneuron/mechanism/mech/mod2c_core_thread.hpp>
#include <coreneuron/sim/scopmath/newton_struct.h>
#include "_kinderiv.h"


namespace coreneuron {
    #ifndef NRN_PRCELLSTATE
    #define NRN_PRCELLSTATE 0
    #endif


    /** channel information */
    static const char *mechanism[] = {
        "6.2.0",
        "pas",
        "g_pas",
        "e_pas",
        0,
        "i_pas",
        0,
        0,
        0
    };


    /** all global variables */
    struct pas_Store {
        int reset;
        int mech_type;
        ThreadDatum* ext_call_thread;
    };

    /** holds object of global variable */
    nrn_pragma_omp(declare target)
    pas_Store pas_global;
    nrn_pragma_acc(declare create (pas_global))
    nrn_pragma_omp(end declare target)


    /** all mechanism instance variables */
    struct pas_Instance  {
        const double* __restrict__ g;
        const double* __restrict__ e;
        double* __restrict__ i;
        double* __restrict__ v_unused;
        double* __restrict__ g_unused;
    };


    /** connect global (scalar) variables to hoc -- */
    static DoubScal hoc_scalar_double[] = {
        0, 0
    };


    /** connect global (array) variables to hoc -- */
    static DoubVec hoc_vector_double[] = {
        0, 0, 0
    };


    static inline int first_pointer_var_index() {
        return -1;
    }


    static inline int float_variables_size() {
        return 5;
    }


    static inline int int_variables_size() {
        return 0;
    }


    static inline int get_mech_type() {
        return pas_global.mech_type;
    }


    static inline Memb_list* get_memb_list(NrnThread* nt) {
        if (nt->_ml_list == NULL) {
            return NULL;
        }
        return nt->_ml_list[get_mech_type()];
    }


    static inline void* mem_alloc(size_t num, size_t size, size_t alignment = 16) {
        void* ptr;
        cudaMallocManaged(&ptr, num*size);
        cudaMemset(ptr, 0, num*size);
        return ptr;
    }


    static inline void mem_free(void* ptr) {
        cudaFree(ptr);
    }


    static inline void coreneuron_abort() {
        printf("Error : Issue while running OpenACC kernel \n");
        assert(0==1);
    }


    /** initialize global variables */
    static inline void setup_global_variables()  {
        static int setup_done = 0;
        if (setup_done) {
            return;
        }
        nrn_pragma_acc(update device (pas_global))
        nrn_pragma_omp(target update to(pas_global))

        setup_done = 1;
    }


    /** free global variables */
    static inline void free_global_variables()  {
        // do nothing
    }


    /** initialize mechanism instance variables */
    static inline void setup_instance(NrnThread* nt, Memb_list* ml)  {
        pas_Instance* inst = (pas_Instance*) mem_alloc(1, sizeof(pas_Instance));
        int pnodecount = ml->_nodecount_padded;
        Datum* indexes = ml->pdata;
        inst->g = nt->compute_gpu ? cnrn_target_deviceptr(ml->data+0*pnodecount) : ml->data+0*pnodecount;
        inst->e = nt->compute_gpu ? cnrn_target_deviceptr(ml->data+1*pnodecount) : ml->data+1*pnodecount;
        inst->i = nt->compute_gpu ? cnrn_target_deviceptr(ml->data+2*pnodecount) : ml->data+2*pnodecount;
        inst->v_unused = nt->compute_gpu ? cnrn_target_deviceptr(ml->data+3*pnodecount) : ml->data+3*pnodecount;
        inst->g_unused = nt->compute_gpu ? cnrn_target_deviceptr(ml->data+4*pnodecount) : ml->data+4*pnodecount;
        ml->instance = inst;
        if(nt->compute_gpu) {
            Memb_list* dml = cnrn_target_deviceptr(ml);
            cnrn_target_memcpy_to_device(&(dml->instance), &(ml->instance));
        }
    }


    /** cleanup mechanism instance variables */
    static inline void cleanup_instance(Memb_list* ml)  {
        pas_Instance* inst = (pas_Instance*) ml->instance;
        mem_free((void*)inst);
    }


    static void nrn_alloc_pas(double* data, Datum* indexes, int type)  {
        // do nothing
    }


    void nrn_constructor_pas(NrnThread* nt, Memb_list* ml, int type) {
        #ifndef CORENEURON_BUILD
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* __restrict__ node_index = ml->nodeindices;
        double* __restrict__ data = ml->data;
        const double* __restrict__ voltage = nt->_actual_v;
        Datum* __restrict__ indexes = ml->pdata;
        ThreadDatum* __restrict__ thread = ml->_thread;
        pas_Instance* __restrict__ inst = (pas_Instance*) ml->instance;

        #endif
    }


    void nrn_destructor_pas(NrnThread* nt, Memb_list* ml, int type) {
        #ifndef CORENEURON_BUILD
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* __restrict__ node_index = ml->nodeindices;
        double* __restrict__ data = ml->data;
        const double* __restrict__ voltage = nt->_actual_v;
        Datum* __restrict__ indexes = ml->pdata;
        ThreadDatum* __restrict__ thread = ml->_thread;
        pas_Instance* __restrict__ inst = (pas_Instance*) ml->instance;

        #endif
    }


    /** initialize channel */
    void nrn_init_pas(NrnThread* nt, Memb_list* ml, int type) {
        nrn_pragma_acc(data present(nt, ml, pas_global) if(nt->compute_gpu))
        {
            int nodecount = ml->nodecount;
            int pnodecount = ml->_nodecount_padded;
            const int* __restrict__ node_index = ml->nodeindices;
            double* __restrict__ data = ml->data;
            const double* __restrict__ voltage = nt->_actual_v;
            Datum* __restrict__ indexes = ml->pdata;
            ThreadDatum* __restrict__ thread = ml->_thread;

            setup_instance(nt, ml);
            pas_Instance* __restrict__ inst = (pas_Instance*) ml->instance;

            nrn_pragma_acc(update device (pas_global))
            nrn_pragma_omp(target update to(pas_global))
            if (_nrn_skip_initmodel == 0) {
                int start = 0;
                int end = nodecount;
                nrn_pragma_acc(parallel loop present(inst, node_index, data, voltage, indexes, thread) async(nt->stream_id) if(nt->compute_gpu))
                nrn_pragma_omp(target teams distribute parallel for is_device_ptr(inst) if(nt->compute_gpu))
                for (int id = start; id < end; id++) {
                    int node_id = node_index[id];
                    double v = voltage[node_id];
                    #if NRN_PRCELLSTATE
                    inst->v_unused[id] = v;
                    #endif
                }
            }
        }
    }


    static inline double nrn_current(int id, int pnodecount, pas_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v) {
        double current = 0.0;
        inst->i[id] = inst->g[id] * (v - inst->e[id]);
        current += inst->i[id];
        return current;
    }


    /** update current */
    void nrn_cur_pas(NrnThread* nt, Memb_list* ml, int type) {
        nrn_pragma_acc(data present(nt, ml, pas_global) if(nt->compute_gpu))
        {
            int nodecount = ml->nodecount;
            int pnodecount = ml->_nodecount_padded;
            const int* __restrict__ node_index = ml->nodeindices;
            double* __restrict__ data = ml->data;
            const double* __restrict__ voltage = nt->_actual_v;
            double* __restrict__  vec_rhs = nt->_actual_rhs;
            double* __restrict__  vec_d = nt->_actual_d;
            Datum* __restrict__ indexes = ml->pdata;
            ThreadDatum* __restrict__ thread = ml->_thread;
            pas_Instance* __restrict__ inst = (pas_Instance*) ml->instance;

            int start = 0;
            int end = nodecount;
            nrn_pragma_acc(parallel loop present(inst, node_index, data, voltage, indexes, thread, vec_rhs, vec_d) async(nt->stream_id) if(nt->compute_gpu))
            nrn_pragma_omp(target teams distribute parallel for is_device_ptr(inst) if(nt->compute_gpu))
            for (int id = start; id < end; id++) {
                int node_id = node_index[id];
                double v = voltage[node_id];
                #if NRN_PRCELLSTATE
                inst->v_unused[id] = v;
                #endif
                double g = nrn_current(id, pnodecount, inst, data, indexes, thread, nt, v+0.001);
                double rhs = nrn_current(id, pnodecount, inst, data, indexes, thread, nt, v);
                g = (g-rhs)/0.001;
                #if NRN_PRCELLSTATE
                inst->g_unused[id] = g;
                #endif
                nrn_pragma_acc(atomic update)
                nrn_pragma_omp(atomic update)
                vec_rhs[node_id] -= rhs;
                nrn_pragma_acc(atomic update)
                nrn_pragma_omp(atomic update)
                vec_d[node_id] += g;
            }
        }
    }


    /** register channel with the simulator */
    void _passive_reg()  {

        int mech_type = nrn_get_mechtype("pas");
        pas_global.mech_type = mech_type;
        if (mech_type == -1) {
            return;
        }

        _nrn_layout_reg(mech_type, 0);
        register_mech(mechanism, nrn_alloc_pas, nrn_cur_pas, NULL, NULL, nrn_init_pas, first_pointer_var_index(), 1);

        setup_global_variables();
        hoc_register_prop_size(mech_type, float_variables_size(), int_variables_size());
        hoc_register_var(hoc_scalar_double, hoc_vector_double, NULL);
    }
}
