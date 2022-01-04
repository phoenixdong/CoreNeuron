/*********************************************************
Model Name      : NetStim
Filename        : netstim.mod
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
#undef DISABLE_OPENACC
#define DISABLE_OPENACC

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
        "NetStim",
        "interval",
        "number",
        "start",
        "noise",
        0,
        0,
        0,
        "donotuse",
        0
    };


    /** all global variables */
    struct NetStim_Store {
        int point_type;
        int reset;
        int mech_type;
        ThreadDatum* ext_call_thread;
    };

    /** holds object of global variable */
    NetStim_Store NetStim_global;


    /** all mechanism instance variables */
    struct NetStim_Instance  {
        const double* __restrict__ interval;
        const double* __restrict__ number;
        const double* __restrict__ start;
        double* __restrict__ noise;
        double* __restrict__ event;
        double* __restrict__ on;
        double* __restrict__ ispike;
        double* __restrict__ v_unused;
        double* __restrict__ tsave;
        const double* __restrict__ node_area;
        void** __restrict__ point_process;
        void** __restrict__ donotuse;
        void** __restrict__ tqitem;
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
        return 2;
    }


    static inline int num_net_receive_args() {
        return 1;
    }


    static inline int float_variables_size() {
        return 9;
    }


    static inline int int_variables_size() {
        return 4;
    }


    static inline int get_mech_type() {
        return NetStim_global.mech_type;
    }


    static inline Memb_list* get_memb_list(NrnThread* nt) {
        if (nt->_ml_list == NULL) {
            return NULL;
        }
        return nt->_ml_list[get_mech_type()];
    }


    static inline void* mem_alloc(size_t num, size_t size, size_t alignment = 16) {
        void* ptr;
        posix_memalign(&ptr, alignment, num*size);
        memset(ptr, 0, size);
        return ptr;
    }


    static inline void mem_free(void* ptr) {
        free(ptr);
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

        setup_done = 1;
    }


    /** free global variables */
    static inline void free_global_variables()  {
        // do nothing
    }


    /** initialize mechanism instance variables */
    static inline void setup_instance(NrnThread* nt, Memb_list* ml)  {
        NetStim_Instance* inst = (NetStim_Instance*) mem_alloc(1, sizeof(NetStim_Instance));
        int pnodecount = ml->_nodecount_padded;
        Datum* indexes = ml->pdata;
        inst->interval = ml->data+0*pnodecount;
        inst->number = ml->data+1*pnodecount;
        inst->start = ml->data+2*pnodecount;
        inst->noise = ml->data+3*pnodecount;
        inst->event = ml->data+4*pnodecount;
        inst->on = ml->data+5*pnodecount;
        inst->ispike = ml->data+6*pnodecount;
        inst->v_unused = ml->data+7*pnodecount;
        inst->tsave = ml->data+8*pnodecount;
        inst->node_area = nt->_data;
        inst->point_process = nt->_vdata;
        inst->donotuse = nt->_vdata;
        inst->tqitem = nt->_vdata;
        ml->instance = inst;
    }


    /** cleanup mechanism instance variables */
    static inline void cleanup_instance(Memb_list* ml)  {
        NetStim_Instance* inst = (NetStim_Instance*) ml->instance;
        mem_free((void*)inst);
    }


    static void nrn_alloc_NetStim(double* data, Datum* indexes, int type)  {
        // do nothing
    }


    void nrn_constructor_NetStim(NrnThread* nt, Memb_list* ml, int type) {
        #ifndef CORENEURON_BUILD
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* __restrict__ node_index = ml->nodeindices;
        double* __restrict__ data = ml->data;
        const double* __restrict__ voltage = nt->_actual_v;
        Datum* __restrict__ indexes = ml->pdata;
        ThreadDatum* __restrict__ thread = ml->_thread;
        NetStim_Instance* __restrict__ inst = (NetStim_Instance*) ml->instance;

        #endif
    }


    void nrn_destructor_NetStim(NrnThread* nt, Memb_list* ml, int type) {
        #ifndef CORENEURON_BUILD
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* __restrict__ node_index = ml->nodeindices;
        double* __restrict__ data = ml->data;
        const double* __restrict__ voltage = nt->_actual_v;
        Datum* __restrict__ indexes = ml->pdata;
        ThreadDatum* __restrict__ thread = ml->_thread;
        NetStim_Instance* __restrict__ inst = (NetStim_Instance*) ml->instance;

        	if (!inst->noise[id]) { return; }
        	if (inst->donotuse[indexes[2*pnodecount + id]]) {
        #if NRNBBCORE
        		{ /* but note that mod2c does not translate DESTRUCTOR */
        #else
        		if (_ran_compat == 2) {
        #endif
        			nrnran123_State** pv = (nrnran123_State**)(&inst->donotuse[indexes[2*pnodecount + id]]);
        			nrnran123_deletestream(*pv);
        			*pv = (nrnran123_State*)0;
        		}
        	}

        #endif
    }
}


using namespace coreneuron;


#if NRNBBCORE /* running in CoreNEURON */
#define IFNEWSTYLE(arg) arg
#else /* running in NEURON */
/*
   1 means noiseFromRandom was called when _ran_compat was previously 0 .
   2 means noiseFromRandom123 was called when _ran_compat was previously 0.
*/
static int _ran_compat; /* specifies the noise style for all instances */
#define IFNEWSTYLE(arg) if(_ran_compat == 2) { arg }
#endif /* running in NEURON */


#include "nrnran123.h"
#if !NRNBBCORE
/* backward compatibility */
double nrn_random_pick(void* r);
void* nrn_random_arg(int argpos);
int nrn_random_isran123(void* r, uint32_t* id1, uint32_t* id2, uint32_t* id3);
int nrn_random123_setseq(void* r, uint32_t seq, char which);
int nrn_random123_getseq(void* r, uint32_t* seq, char* which);
#endif


static void bbcore_write(double* x, int* d, int* xx, int *offset, int id, int pnodecount, double* data, Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v) {
	if (!data[3*pnodecount + id]) { return; }
	/* error if using the legacy scop_exprand */
	if (!nt->_vdata[indexes[2*pnodecount + id]]) {
		fprintf(stderr, "NetStim: cannot use the legacy scop_negexp generator for the random stream.\n");
		assert(0);
	}
	if (d) {
		char which;
		uint32_t* di = ((uint32_t*)d) + *offset;
#if !NRNBBCORE
		if (_ran_compat == 1) {
			void** pv = (void**)(&nt->_vdata[indexes[2*pnodecount + id]]);
			/* error if not using Random123 generator */
			if (!nrn_random_isran123(*pv, di, di+1, di+2)) {
				fprintf(stderr, "NetStim: Random123 generator is required\n");
				assert(0);
			}
			nrn_random123_getseq(*pv, di+3, &which);
			di[4] = (int)which;
		}else{
#else
    {
#endif
			nrnran123_State** pv = (nrnran123_State**)(&nt->_vdata[indexes[2*pnodecount + id]]);
			nrnran123_getids3(*pv, di, di+1, di+2);
			nrnran123_getseq(*pv, di+3, &which);
			di[4] = (int)which;
#if NRNBBCORE
			/* CORENeuron does not call DESTRUCTOR so... */
			nrnran123_deletestream(*pv);
                        *pv = (nrnran123_State*)0;
#endif
		}
		/*printf("Netstim bbcore_write %d %d %d\n", di[0], di[1], di[3]);*/
	}
	*offset += 5;
}
static void bbcore_read(double* x, int* d, int* xx, int* offset, int id, int pnodecount, double* data, Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v) {
	if (!data[3*pnodecount + id]) { return; }
	/* Generally, CoreNEURON, in the context of psolve, begins with
           an empty model so this call takes place in the context of a freshly
           created instance and _p_donotuse is not NULL.
	   However, this function
           is also now called from NEURON at the end of coreneuron psolve
           in order to transfer back the nrnran123 sequence state. That
           allows continuation with a subsequent psolve within NEURON or
           properly transfer back to CoreNEURON if we continue the psolve
           there. So now, extra logic is needed for this call to work in
           a NEURON context.
        */
	uint32_t* di = ((uint32_t*)d) + *offset;
#if NRNBBCORE
	nrnran123_State** pv = (nrnran123_State**)(&nt->_vdata[indexes[2*pnodecount + id]]);
	assert(!nt->_vdata[indexes[2*pnodecount + id]]);
	*pv = nrnran123_newstream3(di[0], di[1], di[2]);
	nrnran123_setseq(*pv, di[3], (char)di[4]);
#else
	uint32_t id1, id2, id3;
	assert(nt->_vdata[indexes[2*pnodecount + id]]);
	if (_ran_compat == 1) { /* Hoc Random.Random123 */
		void** pv = (void**)(&nt->_vdata[indexes[2*pnodecount + id]]);
		int b = nrn_random_isran123(*pv, &id1, &id2, &id3);
		assert(b);
		nrn_random123_setseq(*pv, di[3], (char)di[4]);
	}else{
		assert(_ran_compat == 2);
		nrnran123_State** pv = (nrnran123_State**)(&nt->_vdata[indexes[2*pnodecount + id]]);
		nrnran123_getids3(*pv, &id1, &id2, &id3);
		nrnran123_setseq(*pv, di[3], (char)di[4]);
	}
        /* Random123 on NEURON side has same ids as on CoreNEURON side */
	assert(di[0] == id1 && di[1] == id2 && di[2] == id3);
#endif
	*offset += 5;
}


namespace coreneuron {


    inline double invl_NetStim(int id, int pnodecount, NetStim_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v, double mean);
    inline double erand_NetStim(int id, int pnodecount, NetStim_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v);
    inline double bbsavestate_NetStim(int id, int pnodecount, NetStim_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v);
    inline int seed_NetStim(int id, int pnodecount, NetStim_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v, double x);
    inline int init_sequence_NetStim(int id, int pnodecount, NetStim_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v, double t);
    inline int noiseFromRandom_NetStim(int id, int pnodecount, NetStim_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v);
    inline int noiseFromRandom123_NetStim(int id, int pnodecount, NetStim_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v);
    inline int next_invl_NetStim(int id, int pnodecount, NetStim_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v);


    inline int seed_NetStim(int id, int pnodecount, NetStim_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v, double x) {
        int ret_seed = 0;
        #if !NRNBBCORE

        set_seed(x);
        #endif

        return ret_seed;
    }


    inline int init_sequence_NetStim(int id, int pnodecount, NetStim_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v, double t) {
        int ret_init_sequence = 0;
        if (inst->number[id] > 0.0) {
            inst->on[id] = 1.0;
            inst->event[id] = 0.0;
            inst->ispike[id] = 0.0;
        }
        return ret_init_sequence;
    }


    inline int noiseFromRandom_NetStim(int id, int pnodecount, NetStim_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v) {
        int ret_noiseFromRandom = 0;
        #if !NRNBBCORE
         {
        	void** pv = (void**)(&inst->donotuse[indexes[2*pnodecount + id]]);
        	if (_ran_compat == 2) {
        		fprintf(stderr, "NetStim.noiseFromRandom123 was previously called\n");
        		assert(0);
        	}
        	_ran_compat = 1;
        	if (ifarg(1)) {
        		*pv = nrn_random_arg(1);
        	}else{
        		*pv = (void*)0;
        	}
         }
        #endif

        return ret_noiseFromRandom;
    }


    inline int noiseFromRandom123_NetStim(int id, int pnodecount, NetStim_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v) {
        int ret_noiseFromRandom123 = 0;
        #if !NRNBBCORE
         {
        	nrnran123_State** pv = (nrnran123_State**)(&inst->donotuse[indexes[2*pnodecount + id]]);
        	if (_ran_compat == 1) {
        		fprintf(stderr, "NetStim.noiseFromRandom was previously called\n");
        		assert(0);
        	}
        	_ran_compat = 2;
        	if (*pv) {
        		nrnran123_deletestream(*pv);
        		*pv = (nrnran123_State*)0;
        	}
        	if (ifarg(3)) {
        		*pv = nrnran123_newstream3((uint32_t)*getarg(1), (uint32_t)*getarg(2), (uint32_t)*getarg(3));
        	}else if (ifarg(2)) {
        		*pv = nrnran123_newstream((uint32_t)*getarg(1), (uint32_t)*getarg(2));
        	}
         }
        #endif

        return ret_noiseFromRandom123;
    }


    inline int next_invl_NetStim(int id, int pnodecount, NetStim_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v) {
        int ret_next_invl = 0;
        if (inst->number[id] > 0.0) {
            double invl_in_1;
            {
                double mean_in_1;
                mean_in_1 = inst->interval[id];
                if (mean_in_1 <= 0.0) {
                    mean_in_1 = .01;
                }
                if (inst->noise[id] == 0.0) {
                    invl_in_1 = mean_in_1;
                } else {
                    invl_in_1 = (1.0 - inst->noise[id]) * mean_in_1 + inst->noise[id] * mean_in_1 * erand_NetStim(id, pnodecount, inst, data, indexes, thread, nt, v);
                }
            }
            inst->event[id] = invl_in_1;
        }
        if (inst->ispike[id] >= inst->number[id]) {
            inst->on[id] = 0.0;
        }
        return ret_next_invl;
    }


    inline double invl_NetStim(int id, int pnodecount, NetStim_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v, double mean) {
        double ret_invl = 0.0;
        if (mean <= 0.0) {
            mean = .01;
        }
        if (inst->noise[id] == 0.0) {
            ret_invl = mean;
        } else {
            ret_invl = (1.0 - inst->noise[id]) * mean + inst->noise[id] * mean * erand_NetStim(id, pnodecount, inst, data, indexes, thread, nt, v);
        }
        return ret_invl;
    }


    inline double erand_NetStim(int id, int pnodecount, NetStim_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v) {
        double ret_erand = 0.0;
        	if (inst->donotuse[indexes[2*pnodecount + id]]) {
        		/*
        		:Supports separate independent but reproducible streams for
        		: each instance. However, the corresponding hoc Random
        		: distribution MUST be set to Random.negexp(1)
        		*/
        #if !NRNBBCORE
        		if (_ran_compat == 2) {
        			ret_erand = nrnran123_negexp((nrnran123_State*)inst->donotuse[indexes[2*pnodecount + id]]);
        		}else{
        			ret_erand = nrn_random_pick(inst->donotuse[indexes[2*pnodecount + id]]);
        		}
        #else
        		ret_erand = nrnran123_negexp((nrnran123_State*)inst->donotuse[indexes[2*pnodecount + id]]);
        #endif
        		return ret_erand;
        	}else{
        #if NRNBBCORE
        		assert(0);
        #else
        		/*
        		: the old standby. Cannot use if reproducible parallel sim
        		: independent of nhost or which host this instance is on
        		: is desired, since each instance on this cpu draws from
        		: the same stream
        		*/
        #endif
        	}
        #if !NRNBBCORE

        ret_erand = exprand(1.0);
        #endif

        return ret_erand;
    }


    inline double bbsavestate_NetStim(int id, int pnodecount, NetStim_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v) {
        double ret_bbsavestate = 0.0;
        ret_bbsavestate = 0.0;
        #if !NRNBBCORE
          if (_ran_compat == 2) {
            nrnran123_State** pv = (nrnran123_State**)(&inst->donotuse[indexes[2*pnodecount + id]]);
            if (!*pv) { return 0.0; }
            char which;
            uint32_t seq;
            double *xdir, *xval;
            xdir = hoc_pgetarg(1);
            if (*xdir == -1.) { *xdir = 2; return 0.0; }
            xval = hoc_pgetarg(2);
            if (*xdir == 0.) {
              nrnran123_getseq(*pv, &seq, &which);
              xval[0] = (double)seq;
              xval[1] = (double)which;
            }
            if (*xdir == 1) {
              nrnran123_setseq(*pv, (uint32_t)xval[0], (char)xval[1]);
            }
          } /* else do nothing */
        #endif

        return ret_bbsavestate;
    }


    static inline void net_receive_NetStim(Point_process* pnt, int weight_index, double flag)  {
        int tid = pnt->_tid;
        int id = pnt->_i_instance;
        double v = 0;
        NrnThread* nt = nrn_threads + tid;
        Memb_list* ml = nt->_ml_list[pnt->_type];
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        double* data = ml->data;
        double* weights = nt->weights;
        Datum* indexes = ml->pdata;
        ThreadDatum* thread = ml->_thread;
        NetStim_Instance* inst = (NetStim_Instance*) ml->instance;

        double* w = weights + weight_index + 0;
        double t = nt->_t;
        inst->tsave[id] = t;
        {
            if (flag == 0.0) {
                if ((*w) > 0.0 && inst->on[id] == 0.0) {
                    {
                        double t_in_0;
                        t_in_0 = t;
                        if (inst->number[id] > 0.0) {
                            inst->on[id] = 1.0;
                            inst->event[id] = 0.0;
                            inst->ispike[id] = 0.0;
                        }
                    }
                    {
                        if (inst->number[id] > 0.0) {
                            double invl_in_1;
                            {
                                double mean_in_1;
                                mean_in_1 = inst->interval[id];
                                if (mean_in_1 <= 0.0) {
                                    mean_in_1 = .01;
                                }
                                if (inst->noise[id] == 0.0) {
                                    invl_in_1 = mean_in_1;
                                } else {
                                    invl_in_1 = (1.0 - inst->noise[id]) * mean_in_1 + inst->noise[id] * mean_in_1 * erand_NetStim(id, pnodecount, inst, data, indexes, thread, nt, v);
                                }
                            }
                            inst->event[id] = invl_in_1;
                        }
                        if (inst->ispike[id] >= inst->number[id]) {
                            inst->on[id] = 0.0;
                        }
                    }
                    inst->event[id] = inst->event[id] - inst->interval[id] * (1.0 - inst->noise[id]);
                    artcell_net_send(&inst->tqitem[indexes[3*pnodecount + id]], weight_index, pnt, nt->_t+inst->event[id], 1.0);
                } else if ((*w) < 0.0) {
                    inst->on[id] = 0.0;
                }
            }
            if (flag == 3.0) {
                if (inst->on[id] == 1.0) {
                    {
                        double t_in_1;
                        t_in_1 = t;
                        if (inst->number[id] > 0.0) {
                            inst->on[id] = 1.0;
                            inst->event[id] = 0.0;
                            inst->ispike[id] = 0.0;
                        }
                    }
                    artcell_net_send(&inst->tqitem[indexes[3*pnodecount + id]], weight_index, pnt, nt->_t+0.0, 1.0);
                }
            }
            if (flag == 1.0 && inst->on[id] == 1.0) {
                inst->ispike[id] = inst->ispike[id] + 1.0;
                net_event(pnt, t);
                {
                    if (inst->number[id] > 0.0) {
                        double invl_in_1;
                        {
                            double mean_in_1;
                            mean_in_1 = inst->interval[id];
                            if (mean_in_1 <= 0.0) {
                                mean_in_1 = .01;
                            }
                            if (inst->noise[id] == 0.0) {
                                invl_in_1 = mean_in_1;
                            } else {
                                invl_in_1 = (1.0 - inst->noise[id]) * mean_in_1 + inst->noise[id] * mean_in_1 * erand_NetStim(id, pnodecount, inst, data, indexes, thread, nt, v);
                            }
                        }
                        inst->event[id] = invl_in_1;
                    }
                    if (inst->ispike[id] >= inst->number[id]) {
                        inst->on[id] = 0.0;
                    }
                }
                if (inst->on[id] == 1.0) {
                    artcell_net_send(&inst->tqitem[indexes[3*pnodecount + id]], weight_index, pnt, nt->_t+inst->event[id], 1.0);
                }
            }
        }
    }


    /** initialize channel */
    void nrn_init_NetStim(NrnThread* nt, Memb_list* ml, int type) {
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* __restrict__ node_index = ml->nodeindices;
        double* __restrict__ data = ml->data;
        const double* __restrict__ voltage = nt->_actual_v;
        Datum* __restrict__ indexes = ml->pdata;
        ThreadDatum* __restrict__ thread = ml->_thread;

        setup_instance(nt, ml);
        NetStim_Instance* __restrict__ inst = (NetStim_Instance*) ml->instance;

        if (_nrn_skip_initmodel == 0) {
            int start = 0;
            int end = nodecount;
            for (int id = start; id < end; id++) {
                inst->tsave[id] = -1e20;
                double v = 0.0;
                	  if (inst->donotuse[indexes[2*pnodecount + id]]) {
                	    /* only this style initializes the stream on finitialize */
                	    IFNEWSTYLE(nrnran123_setseq((nrnran123_State*)inst->donotuse[indexes[2*pnodecount + id]], 0, 0);)
                	  }
                	

                inst->on[id] = 0.0;
                inst->ispike[id] = 0.0;
                if (inst->noise[id] < 0.0) {
                    inst->noise[id] = 0.0;
                }
                if (inst->noise[id] > 1.0) {
                    inst->noise[id] = 1.0;
                }
                if (inst->start[id] >= 0.0 && inst->number[id] > 0.0) {
                    double invl_in_0;
                    inst->on[id] = 1.0;
                    {
                        double mean_in_0;
                        mean_in_0 = inst->interval[id];
                        if (mean_in_0 <= 0.0) {
                            mean_in_0 = .01;
                        }
                        if (inst->noise[id] == 0.0) {
                            invl_in_0 = mean_in_0;
                        } else {
                            invl_in_0 = (1.0 - inst->noise[id]) * mean_in_0 + inst->noise[id] * mean_in_0 * erand_NetStim(id, pnodecount, inst, data, indexes, thread, nt, v);
                        }
                    }
                    inst->event[id] = inst->start[id] + invl_in_0 - inst->interval[id] * (1.0 - inst->noise[id]);
                    if (inst->event[id] < 0.0) {
                        inst->event[id] = 0.0;
                    }
                    artcell_net_send(&inst->tqitem[indexes[3*pnodecount + id]], 0, (Point_process*)inst->point_process[indexes[1*pnodecount + id]], nt->_t+inst->event[id], 3.0);
                }
            }
        }
    }


    /** register channel with the simulator */
    void _netstim_reg()  {

        int mech_type = nrn_get_mechtype("NetStim");
        NetStim_global.mech_type = mech_type;
        if (mech_type == -1) {
            return;
        }

        _nrn_layout_reg(mech_type, 0);
        point_register_mech(mechanism, nrn_alloc_NetStim, NULL, NULL, NULL, nrn_init_NetStim, first_pointer_var_index(), NULL, nrn_destructor_NetStim, 1);

        setup_global_variables();
        hoc_reg_bbcore_read(mech_type, bbcore_read);
        hoc_reg_bbcore_write(mech_type, bbcore_write);
        hoc_register_prop_size(mech_type, float_variables_size(), int_variables_size());
        hoc_register_dparam_semantics(mech_type, 0, "area");
        hoc_register_dparam_semantics(mech_type, 1, "pntproc");
        hoc_register_dparam_semantics(mech_type, 2, "bbcorepointer");
        hoc_register_dparam_semantics(mech_type, 3, "netsend");
        add_nrn_has_net_event(mech_type);
        add_nrn_artcell(mech_type, 3);
        set_pnt_receive(mech_type, net_receive_NetStim, nullptr, num_net_receive_args());
        hoc_register_net_send_buffering(mech_type);
        hoc_register_var(hoc_scalar_double, hoc_vector_double, NULL);
    }
}
