/*********************************************************
Model Name      : hh
Filename        : hh.mod
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
        "hh",
        "gnabar_hh",
        "gkbar_hh",
        "gl_hh",
        "el_hh",
        0,
        "gna_hh",
        "gk_hh",
        "il_hh",
        "minf_hh",
        "hinf_hh",
        "ninf_hh",
        "mtau_hh",
        "htau_hh",
        "ntau_hh",
        0,
        "m_hh",
        "h_hh",
        "n_hh",
        0,
        0
    };


    /** all global variables */
    struct hh_Store {
        int na_type;
        int k_type;
        double m0;
        double h0;
        double n0;
        int reset;
        int mech_type;
        int* slist1;
        int* dlist1;
        ThreadDatum* ext_call_thread;
    };

    /** holds object of global variable */
    nrn_pragma_omp(declare target)
    hh_Store hh_global;
    nrn_pragma_acc(declare create (hh_global))
    nrn_pragma_omp(end declare target)


    /** all mechanism instance variables */
    struct hh_Instance  {
        const double* __restrict__ gnabar;
        const double* __restrict__ gkbar;
        const double* __restrict__ gl;
        const double* __restrict__ el;
        double* __restrict__ gna;
        double* __restrict__ gk;
        double* __restrict__ il;
        double* __restrict__ minf;
        double* __restrict__ hinf;
        double* __restrict__ ninf;
        double* __restrict__ mtau;
        double* __restrict__ htau;
        double* __restrict__ ntau;
        double* __restrict__ m;
        double* __restrict__ h;
        double* __restrict__ n;
        double* __restrict__ Dm;
        double* __restrict__ Dh;
        double* __restrict__ Dn;
        double* __restrict__ ena;
        double* __restrict__ ek;
        double* __restrict__ ina;
        double* __restrict__ ik;
        double* __restrict__ v_unused;
        double* __restrict__ g_unused;
        const double* __restrict__ ion_ena;
        double* __restrict__ ion_ina;
        double* __restrict__ ion_dinadv;
        const double* __restrict__ ion_ek;
        double* __restrict__ ion_ik;
        double* __restrict__ ion_dikdv;
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
        return 25;
    }


    static inline int int_variables_size() {
        return 6;
    }


    static inline int get_mech_type() {
        return hh_global.mech_type;
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
        hh_global.slist1 = (int*) mem_alloc(3, sizeof(int));
        hh_global.dlist1 = (int*) mem_alloc(3, sizeof(int));
        hh_global.slist1[0] = 13;
        hh_global.dlist1[0] = 16;
        hh_global.slist1[1] = 14;
        hh_global.dlist1[1] = 17;
        hh_global.slist1[2] = 15;
        hh_global.dlist1[2] = 18;
        hh_global.m0 = 0.0;
        hh_global.h0 = 0.0;
        hh_global.n0 = 0.0;
        nrn_pragma_acc(update device (hh_global))
        nrn_pragma_omp(target update to(hh_global))

        setup_done = 1;
    }


    /** free global variables */
    static inline void free_global_variables()  {
        mem_free(hh_global.slist1);
        mem_free(hh_global.dlist1);
    }


    /** initialize mechanism instance variables */
    static inline void setup_instance(NrnThread* nt, Memb_list* ml)  {
        hh_Instance* inst = (hh_Instance*) mem_alloc(1, sizeof(hh_Instance));
        int pnodecount = ml->_nodecount_padded;
        Datum* indexes = ml->pdata;
        inst->gnabar = nt->compute_gpu ? cnrn_target_deviceptr(ml->data+0*pnodecount) : ml->data+0*pnodecount;
        inst->gkbar = nt->compute_gpu ? cnrn_target_deviceptr(ml->data+1*pnodecount) : ml->data+1*pnodecount;
        inst->gl = nt->compute_gpu ? cnrn_target_deviceptr(ml->data+2*pnodecount) : ml->data+2*pnodecount;
        inst->el = nt->compute_gpu ? cnrn_target_deviceptr(ml->data+3*pnodecount) : ml->data+3*pnodecount;
        inst->gna = nt->compute_gpu ? cnrn_target_deviceptr(ml->data+4*pnodecount) : ml->data+4*pnodecount;
        inst->gk = nt->compute_gpu ? cnrn_target_deviceptr(ml->data+5*pnodecount) : ml->data+5*pnodecount;
        inst->il = nt->compute_gpu ? cnrn_target_deviceptr(ml->data+6*pnodecount) : ml->data+6*pnodecount;
        inst->minf = nt->compute_gpu ? cnrn_target_deviceptr(ml->data+7*pnodecount) : ml->data+7*pnodecount;
        inst->hinf = nt->compute_gpu ? cnrn_target_deviceptr(ml->data+8*pnodecount) : ml->data+8*pnodecount;
        inst->ninf = nt->compute_gpu ? cnrn_target_deviceptr(ml->data+9*pnodecount) : ml->data+9*pnodecount;
        inst->mtau = nt->compute_gpu ? cnrn_target_deviceptr(ml->data+10*pnodecount) : ml->data+10*pnodecount;
        inst->htau = nt->compute_gpu ? cnrn_target_deviceptr(ml->data+11*pnodecount) : ml->data+11*pnodecount;
        inst->ntau = nt->compute_gpu ? cnrn_target_deviceptr(ml->data+12*pnodecount) : ml->data+12*pnodecount;
        inst->m = nt->compute_gpu ? cnrn_target_deviceptr(ml->data+13*pnodecount) : ml->data+13*pnodecount;
        inst->h = nt->compute_gpu ? cnrn_target_deviceptr(ml->data+14*pnodecount) : ml->data+14*pnodecount;
        inst->n = nt->compute_gpu ? cnrn_target_deviceptr(ml->data+15*pnodecount) : ml->data+15*pnodecount;
        inst->Dm = nt->compute_gpu ? cnrn_target_deviceptr(ml->data+16*pnodecount) : ml->data+16*pnodecount;
        inst->Dh = nt->compute_gpu ? cnrn_target_deviceptr(ml->data+17*pnodecount) : ml->data+17*pnodecount;
        inst->Dn = nt->compute_gpu ? cnrn_target_deviceptr(ml->data+18*pnodecount) : ml->data+18*pnodecount;
        inst->ena = nt->compute_gpu ? cnrn_target_deviceptr(ml->data+19*pnodecount) : ml->data+19*pnodecount;
        inst->ek = nt->compute_gpu ? cnrn_target_deviceptr(ml->data+20*pnodecount) : ml->data+20*pnodecount;
        inst->ina = nt->compute_gpu ? cnrn_target_deviceptr(ml->data+21*pnodecount) : ml->data+21*pnodecount;
        inst->ik = nt->compute_gpu ? cnrn_target_deviceptr(ml->data+22*pnodecount) : ml->data+22*pnodecount;
        inst->v_unused = nt->compute_gpu ? cnrn_target_deviceptr(ml->data+23*pnodecount) : ml->data+23*pnodecount;
        inst->g_unused = nt->compute_gpu ? cnrn_target_deviceptr(ml->data+24*pnodecount) : ml->data+24*pnodecount;
        inst->ion_ena = nt->compute_gpu ? cnrn_target_deviceptr(nt->_data) : nt->_data;
        inst->ion_ina = nt->compute_gpu ? cnrn_target_deviceptr(nt->_data) : nt->_data;
        inst->ion_dinadv = nt->compute_gpu ? cnrn_target_deviceptr(nt->_data) : nt->_data;
        inst->ion_ek = nt->compute_gpu ? cnrn_target_deviceptr(nt->_data) : nt->_data;
        inst->ion_ik = nt->compute_gpu ? cnrn_target_deviceptr(nt->_data) : nt->_data;
        inst->ion_dikdv = nt->compute_gpu ? cnrn_target_deviceptr(nt->_data) : nt->_data;
        ml->instance = inst;
        if(nt->compute_gpu) {
            Memb_list* dml = cnrn_target_deviceptr(ml);
            cnrn_target_memcpy_to_device(&(dml->instance), &(ml->instance));
        }
    }


    /** cleanup mechanism instance variables */
    static inline void cleanup_instance(Memb_list* ml)  {
        hh_Instance* inst = (hh_Instance*) ml->instance;
        mem_free((void*)inst);
    }


    static void nrn_alloc_hh(double* data, Datum* indexes, int type)  {
        // do nothing
    }


    void nrn_constructor_hh(NrnThread* nt, Memb_list* ml, int type) {
        #ifndef CORENEURON_BUILD
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* __restrict__ node_index = ml->nodeindices;
        double* __restrict__ data = ml->data;
        const double* __restrict__ voltage = nt->_actual_v;
        Datum* __restrict__ indexes = ml->pdata;
        ThreadDatum* __restrict__ thread = ml->_thread;
        hh_Instance* __restrict__ inst = (hh_Instance*) ml->instance;

        #endif
    }


    void nrn_destructor_hh(NrnThread* nt, Memb_list* ml, int type) {
        #ifndef CORENEURON_BUILD
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* __restrict__ node_index = ml->nodeindices;
        double* __restrict__ data = ml->data;
        const double* __restrict__ voltage = nt->_actual_v;
        Datum* __restrict__ indexes = ml->pdata;
        ThreadDatum* __restrict__ thread = ml->_thread;
        hh_Instance* __restrict__ inst = (hh_Instance*) ml->instance;

        #endif
    }


    inline double vtrap_hh(int id, int pnodecount, hh_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v, double x, double y);
    inline int rates_hh(int id, int pnodecount, hh_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v, double arg_v);


    inline int rates_hh(int id, int pnodecount, hh_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v, double arg_v) {
        int ret_rates = 0;
        double alpha, beta, sum, q10, vtrap_in_0, vtrap_in_1;
        q10 = pow(3.0, ((celsius - 6.3) / 10.0));
        {
            double x_in_0, y_in_0;
            x_in_0 =  -(arg_v + 40.0);
            y_in_0 = 10.0;
            if (fabs(x_in_0 / y_in_0) < 1e-6) {
                vtrap_in_0 = y_in_0 * (1.0 - x_in_0 / y_in_0 / 2.0);
            } else {
                vtrap_in_0 = x_in_0 / (exp(x_in_0 / y_in_0) - 1.0);
            }
        }
        alpha = .1 * vtrap_in_0;
        beta = 4.0 * exp( -(arg_v + 65.0) / 18.0);
        sum = alpha + beta;
        inst->mtau[id] = 1.0 / (q10 * sum);
        inst->minf[id] = alpha / sum;
        alpha = .07 * exp( -(arg_v + 65.0) / 20.0);
        beta = 1.0 / (exp( -(arg_v + 35.0) / 10.0) + 1.0);
        sum = alpha + beta;
        inst->htau[id] = 1.0 / (q10 * sum);
        inst->hinf[id] = alpha / sum;
        {
            double x_in_1, y_in_1;
            x_in_1 =  -(arg_v + 55.0);
            y_in_1 = 10.0;
            if (fabs(x_in_1 / y_in_1) < 1e-6) {
                vtrap_in_1 = y_in_1 * (1.0 - x_in_1 / y_in_1 / 2.0);
            } else {
                vtrap_in_1 = x_in_1 / (exp(x_in_1 / y_in_1) - 1.0);
            }
        }
        alpha = .01 * vtrap_in_1;
        beta = .125 * exp( -(arg_v + 65.0) / 80.0);
        sum = alpha + beta;
        inst->ntau[id] = 1.0 / (q10 * sum);
        inst->ninf[id] = alpha / sum;
        return ret_rates;
    }


    inline double vtrap_hh(int id, int pnodecount, hh_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v, double x, double y) {
        double ret_vtrap = 0.0;
        if (fabs(x / y) < 1e-6) {
            ret_vtrap = y * (1.0 - x / y / 2.0);
        } else {
            ret_vtrap = x / (exp(x / y) - 1.0);
        }
        return ret_vtrap;
    }


    /** initialize channel */
    void nrn_init_hh(NrnThread* nt, Memb_list* ml, int type) {
        nrn_pragma_acc(data present(nt, ml, hh_global) if(nt->compute_gpu))
        {
            int nodecount = ml->nodecount;
            int pnodecount = ml->_nodecount_padded;
            const int* __restrict__ node_index = ml->nodeindices;
            double* __restrict__ data = ml->data;
            const double* __restrict__ voltage = nt->_actual_v;
            Datum* __restrict__ indexes = ml->pdata;
            ThreadDatum* __restrict__ thread = ml->_thread;

            setup_instance(nt, ml);
            hh_Instance* __restrict__ inst = (hh_Instance*) ml->instance;

            nrn_pragma_acc(update device (hh_global))
            nrn_pragma_omp(target update to(hh_global))
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
                    inst->ena[id] = inst->ion_ena[indexes[0*pnodecount + id]];
                    inst->ek[id] = inst->ion_ek[indexes[3*pnodecount + id]];
                    inst->m[id] = hh_global.m0;
                    inst->h[id] = hh_global.h0;
                    inst->n[id] = hh_global.n0;
                    {
                        double alpha, beta, sum, q10, vtrap_in_0, vtrap_in_1, v_in_0;
                        v_in_0 = v;
                        q10 = pow(3.0, ((celsius - 6.3) / 10.0));
                        {
                            double x_in_0, y_in_0;
                            x_in_0 =  -(v_in_0 + 40.0);
                            y_in_0 = 10.0;
                            if (fabs(x_in_0 / y_in_0) < 1e-6) {
                                vtrap_in_0 = y_in_0 * (1.0 - x_in_0 / y_in_0 / 2.0);
                            } else {
                                vtrap_in_0 = x_in_0 / (exp(x_in_0 / y_in_0) - 1.0);
                            }
                        }
                        alpha = .1 * vtrap_in_0;
                        beta = 4.0 * exp( -(v_in_0 + 65.0) / 18.0);
                        sum = alpha + beta;
                        inst->mtau[id] = 1.0 / (q10 * sum);
                        inst->minf[id] = alpha / sum;
                        alpha = .07 * exp( -(v_in_0 + 65.0) / 20.0);
                        beta = 1.0 / (exp( -(v_in_0 + 35.0) / 10.0) + 1.0);
                        sum = alpha + beta;
                        inst->htau[id] = 1.0 / (q10 * sum);
                        inst->hinf[id] = alpha / sum;
                        {
                            double x_in_1, y_in_1;
                            x_in_1 =  -(v_in_0 + 55.0);
                            y_in_1 = 10.0;
                            if (fabs(x_in_1 / y_in_1) < 1e-6) {
                                vtrap_in_1 = y_in_1 * (1.0 - x_in_1 / y_in_1 / 2.0);
                            } else {
                                vtrap_in_1 = x_in_1 / (exp(x_in_1 / y_in_1) - 1.0);
                            }
                        }
                        alpha = .01 * vtrap_in_1;
                        beta = .125 * exp( -(v_in_0 + 65.0) / 80.0);
                        sum = alpha + beta;
                        inst->ntau[id] = 1.0 / (q10 * sum);
                        inst->ninf[id] = alpha / sum;
                    }
                    inst->m[id] = inst->minf[id];
                    inst->h[id] = inst->hinf[id];
                    inst->n[id] = inst->ninf[id];
                }
            }
        }
    }


    static inline double nrn_current(int id, int pnodecount, hh_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v) {
        double current = 0.0;
        inst->gna[id] = inst->gnabar[id] * inst->m[id] * inst->m[id] * inst->m[id] * inst->h[id];
        inst->ina[id] = inst->gna[id] * (v - inst->ena[id]);
        inst->gk[id] = inst->gkbar[id] * inst->n[id] * inst->n[id] * inst->n[id] * inst->n[id];
        inst->ik[id] = inst->gk[id] * (v - inst->ek[id]);
        inst->il[id] = inst->gl[id] * (v - inst->el[id]);
        current += inst->il[id];
        current += inst->ina[id];
        current += inst->ik[id];
        return current;
    }


    /** update current */
    void nrn_cur_hh(NrnThread* nt, Memb_list* ml, int type) {
        nrn_pragma_acc(data present(nt, ml, hh_global) if(nt->compute_gpu))
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
            hh_Instance* __restrict__ inst = (hh_Instance*) ml->instance;

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
                inst->ena[id] = inst->ion_ena[indexes[0*pnodecount + id]];
                inst->ek[id] = inst->ion_ek[indexes[3*pnodecount + id]];
                double g = nrn_current(id, pnodecount, inst, data, indexes, thread, nt, v+0.001);
                double dina = inst->ina[id];
                double dik = inst->ik[id];
                double rhs = nrn_current(id, pnodecount, inst, data, indexes, thread, nt, v);
                g = (g-rhs)/0.001;
                inst->ion_dinadv[indexes[2*pnodecount + id]] += (dina-inst->ina[id])/0.001;
                inst->ion_dikdv[indexes[5*pnodecount + id]] += (dik-inst->ik[id])/0.001;
                inst->ion_ina[indexes[1*pnodecount + id]] += inst->ina[id];
                inst->ion_ik[indexes[4*pnodecount + id]] += inst->ik[id];
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


    /** update state */
    void nrn_state_hh(NrnThread* nt, Memb_list* ml, int type) {
        nrn_pragma_acc(data present(nt, ml, hh_global) if(nt->compute_gpu))
        {
            int nodecount = ml->nodecount;
            int pnodecount = ml->_nodecount_padded;
            const int* __restrict__ node_index = ml->nodeindices;
            double* __restrict__ data = ml->data;
            const double* __restrict__ voltage = nt->_actual_v;
            Datum* __restrict__ indexes = ml->pdata;
            ThreadDatum* __restrict__ thread = ml->_thread;
            hh_Instance* __restrict__ inst = (hh_Instance*) ml->instance;

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
                inst->ena[id] = inst->ion_ena[indexes[0*pnodecount + id]];
                inst->ek[id] = inst->ion_ek[indexes[3*pnodecount + id]];
                {
                    double alpha, beta, sum, q10, vtrap_in_0, vtrap_in_1, v_in_1;
                    v_in_1 = v;
                    q10 = pow(3.0, ((celsius - 6.3) / 10.0));
                    {
                        double x_in_0, y_in_0;
                        x_in_0 =  -(v_in_1 + 40.0);
                        y_in_0 = 10.0;
                        if (fabs(x_in_0 / y_in_0) < 1e-6) {
                            vtrap_in_0 = y_in_0 * (1.0 - x_in_0 / y_in_0 / 2.0);
                        } else {
                            vtrap_in_0 = x_in_0 / (exp(x_in_0 / y_in_0) - 1.0);
                        }
                    }
                    alpha = .1 * vtrap_in_0;
                    beta = 4.0 * exp( -(v_in_1 + 65.0) / 18.0);
                    sum = alpha + beta;
                    inst->mtau[id] = 1.0 / (q10 * sum);
                    inst->minf[id] = alpha / sum;
                    alpha = .07 * exp( -(v_in_1 + 65.0) / 20.0);
                    beta = 1.0 / (exp( -(v_in_1 + 35.0) / 10.0) + 1.0);
                    sum = alpha + beta;
                    inst->htau[id] = 1.0 / (q10 * sum);
                    inst->hinf[id] = alpha / sum;
                    {
                        double x_in_1, y_in_1;
                        x_in_1 =  -(v_in_1 + 55.0);
                        y_in_1 = 10.0;
                        if (fabs(x_in_1 / y_in_1) < 1e-6) {
                            vtrap_in_1 = y_in_1 * (1.0 - x_in_1 / y_in_1 / 2.0);
                        } else {
                            vtrap_in_1 = x_in_1 / (exp(x_in_1 / y_in_1) - 1.0);
                        }
                    }
                    alpha = .01 * vtrap_in_1;
                    beta = .125 * exp( -(v_in_1 + 65.0) / 80.0);
                    sum = alpha + beta;
                    inst->ntau[id] = 1.0 / (q10 * sum);
                    inst->ninf[id] = alpha / sum;
                }
                inst->m[id] = inst->m[id] + (1.0 - exp(nt->_dt * (((( -1.0))) / inst->mtau[id]))) * ( -(((inst->minf[id])) / inst->mtau[id]) / (((( -1.0))) / inst->mtau[id]) - inst->m[id]);
                inst->h[id] = inst->h[id] + (1.0 - exp(nt->_dt * (((( -1.0))) / inst->htau[id]))) * ( -(((inst->hinf[id])) / inst->htau[id]) / (((( -1.0))) / inst->htau[id]) - inst->h[id]);
                inst->n[id] = inst->n[id] + (1.0 - exp(nt->_dt * (((( -1.0))) / inst->ntau[id]))) * ( -(((inst->ninf[id])) / inst->ntau[id]) / (((( -1.0))) / inst->ntau[id]) - inst->n[id]);
            }
        }
    }


    /** register channel with the simulator */
    void _hh_reg()  {

        int mech_type = nrn_get_mechtype("hh");
        hh_global.mech_type = mech_type;
        if (mech_type == -1) {
            return;
        }

        _nrn_layout_reg(mech_type, 0);
        register_mech(mechanism, nrn_alloc_hh, nrn_cur_hh, NULL, nrn_state_hh, nrn_init_hh, first_pointer_var_index(), 1);
        hh_global.na_type = nrn_get_mechtype("na_ion");
        hh_global.k_type = nrn_get_mechtype("k_ion");

        setup_global_variables();
        hoc_register_prop_size(mech_type, float_variables_size(), int_variables_size());
        hoc_register_dparam_semantics(mech_type, 0, "na_ion");
        hoc_register_dparam_semantics(mech_type, 1, "na_ion");
        hoc_register_dparam_semantics(mech_type, 2, "na_ion");
        hoc_register_dparam_semantics(mech_type, 3, "k_ion");
        hoc_register_dparam_semantics(mech_type, 4, "k_ion");
        hoc_register_dparam_semantics(mech_type, 5, "k_ion");
        hoc_register_var(hoc_scalar_double, hoc_vector_double, NULL);
    }
}
