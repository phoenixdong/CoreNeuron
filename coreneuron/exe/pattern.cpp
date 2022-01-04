/*********************************************************
Model Name      : PatternStim
Filename        : pattern.mod
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
        "PatternStim",
        "fake_output",
        0,
        0,
        0,
        "ptr",
        0
    };


    /** all global variables */
    struct PatternStim_Store {
        int point_type;
        int reset;
        int mech_type;
        ThreadDatum* ext_call_thread;
    };

    /** holds object of global variable */
    PatternStim_Store PatternStim_global;


    /** all mechanism instance variables */
    struct PatternStim_Instance  {
        double* __restrict__ fake_output;
        double* __restrict__ v_unused;
        double* __restrict__ tsave;
        const double* __restrict__ node_area;
        void** __restrict__ point_process;
        void** __restrict__ ptr;
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
        return 3;
    }


    static inline int int_variables_size() {
        return 4;
    }


    static inline int get_mech_type() {
        return PatternStim_global.mech_type;
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
        PatternStim_Instance* inst = (PatternStim_Instance*) mem_alloc(1, sizeof(PatternStim_Instance));
        int pnodecount = ml->_nodecount_padded;
        Datum* indexes = ml->pdata;
        inst->fake_output = ml->data+0*pnodecount;
        inst->v_unused = ml->data+1*pnodecount;
        inst->tsave = ml->data+2*pnodecount;
        inst->node_area = nt->_data;
        inst->point_process = nt->_vdata;
        inst->ptr = nt->_vdata;
        inst->tqitem = nt->_vdata;
        ml->instance = inst;
    }


    /** cleanup mechanism instance variables */
    static inline void cleanup_instance(Memb_list* ml)  {
        PatternStim_Instance* inst = (PatternStim_Instance*) ml->instance;
        mem_free((void*)inst);
    }


    static void nrn_alloc_PatternStim(double* data, Datum* indexes, int type)  {
        // do nothing
    }


    void nrn_constructor_PatternStim(NrnThread* nt, Memb_list* ml, int type) {
        #ifndef CORENEURON_BUILD
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* __restrict__ node_index = ml->nodeindices;
        double* __restrict__ data = ml->data;
        const double* __restrict__ voltage = nt->_actual_v;
        Datum* __restrict__ indexes = ml->pdata;
        ThreadDatum* __restrict__ thread = ml->_thread;
        PatternStim_Instance* __restrict__ inst = (PatternStim_Instance*) ml->instance;

        #endif
    }


    void nrn_destructor_PatternStim(NrnThread* nt, Memb_list* ml, int type) {
        #ifndef CORENEURON_BUILD
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* __restrict__ node_index = ml->nodeindices;
        double* __restrict__ data = ml->data;
        const double* __restrict__ voltage = nt->_actual_v;
        Datum* __restrict__ indexes = ml->pdata;
        ThreadDatum* __restrict__ thread = ml->_thread;
        PatternStim_Instance* __restrict__ inst = (PatternStim_Instance*) ml->instance;

        #endif
    }
}


using namespace coreneuron;


struct Info {
	int size;
	double* tvec;
	int* gidvec;
	int index;
};
#define INFOCAST Info** ip = (Info**)(&(nt->_vdata[indexes[2*pnodecount + id]]))


Info* mkinfo(int id, int pnodecount, double* data, Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v) {
	INFOCAST;
	Info* info = (Info*)hoc_Emalloc(sizeof(Info)); hoc_malchk();
	info->size = 0;
	info->tvec = nullptr;
	info->gidvec = nullptr;
	info->index = 0;
	return info;
}
/* for CoreNEURON checkpoint save and restore */
namespace coreneuron {
int checkpoint_save_patternstim(int id, int pnodecount, double* data, Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v) {
	INFOCAST; Info* info = *ip;
	return info->index;
}
void checkpoint_restore_patternstim(int _index, double _te, int id, int pnodecount, double* data, Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v) {
    INFOCAST; Info* info = *ip;
    info->index = _index;
    artcell_net_send(&nt->_vdata[indexes[3*pnodecount + id]], -1, (Point_process*)nt->_vdata[indexes[1*pnodecount+id]], _te, 1.0);
}
} 


static void bbcore_write(double* x, int* d, int* xx, int *offset, int id, int pnodecount, double* data, Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v){}
static void bbcore_read(double* x, int* d, int* xx, int* offset, int id, int pnodecount, double* data, Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v){}
namespace coreneuron {
void pattern_stim_setup_helper(int size, double* tv, int* gv, int id, int pnodecount, double* data, Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v) {
	INFOCAST;
	Info* info = mkinfo(id, pnodecount, data, indexes, thread, nt, v);
	*ip = info;
	info->size = size;
	info->tvec = tv;
	info->gidvec = gv;
	artcell_net_send ( &nt->_vdata[indexes[3*pnodecount + id]], -1, (Point_process*) nt->_vdata[indexes[1*pnodecount+id]], nt->_t +  0.0 , 1.0 ) ;
}
Info** pattern_stim_info_ref(int id, int pnodecount, double* data, Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v) {
    INFOCAST;
    return ip; 
}
} 


namespace coreneuron {


    inline double initps_PatternStim(int id, int pnodecount, PatternStim_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v);
    inline double sendgroup_PatternStim(int id, int pnodecount, PatternStim_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v);


    inline double initps_PatternStim(int id, int pnodecount, PatternStim_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v) {
        double ret_initps = 0.0;
         {
        	INFOCAST; Info* info = *ip;
        	info->index = 0;
        	if (info && info->tvec) {
        		ret_initps = 1.;
        	}else{
        		ret_initps = 0.;
        	}
        }

        return ret_initps;
    }


    inline double sendgroup_PatternStim(int id, int pnodecount, PatternStim_Instance* inst, double* data, const Datum* indexes, ThreadDatum* thread, NrnThread* nt, double v) {
        double ret_sendgroup = 0.0;
         {
        	INFOCAST; Info* info = *ip;
        	int size = info->size;
        	int fake_out;
        	double* tvec = info->tvec;
        	int* gidvec = info->gidvec;
        	int i;
        	fake_out = inst->fake_output[id] ? 1 : 0;
        	for (i=0; info->index < size; ++i) {
        		/* only if the gid is NOT on this machine */
        		nrn_fake_fire(gidvec[info->index], tvec[info->index], fake_out);
        		++info->index;
        		if (i > 100 && nt->_t < tvec[info->index]) { break; }
        	}
        	if (info->index >= size) {
        		ret_sendgroup = nt->_t - 1.;
        	}else{
        		ret_sendgroup = tvec[info->index];
        	}
        }

        return ret_sendgroup;
    }


    static inline void net_receive_PatternStim(Point_process* pnt, int weight_index, double flag)  {
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
        PatternStim_Instance* inst = (PatternStim_Instance*) ml->instance;

        double t = nt->_t;
        inst->tsave[id] = t;
        {
            double nst;
            if (flag == 1.0) {
                double sendgroup_in_0;
                {
                     {
                    	INFOCAST; Info* info = *ip;
                    	int size = info->size;
                    	int fake_out;
                    	double* tvec = info->tvec;
                    	int* gidvec = info->gidvec;
                    	int i;
                    	fake_out = inst->fake_output[id] ? 1 : 0;
                    	for (i=0; info->index < size; ++i) {
                    		/* only if the gid is NOT on this machine */
                    		nrn_fake_fire(gidvec[info->index], tvec[info->index], fake_out);
                    		++info->index;
                    		if (i > 100 && t < tvec[info->index]) { break; }
                    	}
                    	if (info->index >= size) {
                    		sendgroup_in_0 = t - 1.;
                    	}else{
                    		sendgroup_in_0 = tvec[info->index];
                    	}
                    }

                }
                nst = sendgroup_in_0;
                if (nst >= t) {
                    artcell_net_send(&inst->tqitem[indexes[3*pnodecount + id]], weight_index, pnt, nt->_t+nst - t, 1.0);
                }
            }
        }
    }


    /** initialize channel */
    void nrn_init_PatternStim(NrnThread* nt, Memb_list* ml, int type) {
        int nodecount = ml->nodecount;
        int pnodecount = ml->_nodecount_padded;
        const int* __restrict__ node_index = ml->nodeindices;
        double* __restrict__ data = ml->data;
        const double* __restrict__ voltage = nt->_actual_v;
        Datum* __restrict__ indexes = ml->pdata;
        ThreadDatum* __restrict__ thread = ml->_thread;

        setup_instance(nt, ml);
        PatternStim_Instance* __restrict__ inst = (PatternStim_Instance*) ml->instance;

        if (_nrn_skip_initmodel == 0) {
            int start = 0;
            int end = nodecount;
            for (int id = start; id < end; id++) {
                inst->tsave[id] = -1e20;
                double v = 0.0;
                double initps_in_0;
                {
                     {
                    	INFOCAST; Info* info = *ip;
                    	info->index = 0;
                    	if (info && info->tvec) {
                    		initps_in_0 = 1.;
                    	}else{
                    		initps_in_0 = 0.;
                    	}
                    }

                }
                if (initps_in_0 > 0.0) {
                    artcell_net_send(&inst->tqitem[indexes[3*pnodecount + id]], 0, (Point_process*)inst->point_process[indexes[1*pnodecount + id]], nt->_t+0.0, 1.0);
                }
            }
        }
    }


    /** register channel with the simulator */
    void _pattern_reg()  {

        int mech_type = nrn_get_mechtype("PatternStim");
        PatternStim_global.mech_type = mech_type;
        if (mech_type == -1) {
            return;
        }

        _nrn_layout_reg(mech_type, 0);
        point_register_mech(mechanism, nrn_alloc_PatternStim, NULL, NULL, NULL, nrn_init_PatternStim, first_pointer_var_index(), NULL, NULL, 1);

        setup_global_variables();
        hoc_reg_bbcore_read(mech_type, bbcore_read);
        hoc_reg_bbcore_write(mech_type, bbcore_write);
        hoc_register_prop_size(mech_type, float_variables_size(), int_variables_size());
        hoc_register_dparam_semantics(mech_type, 0, "area");
        hoc_register_dparam_semantics(mech_type, 1, "pntproc");
        hoc_register_dparam_semantics(mech_type, 2, "bbcorepointer");
        hoc_register_dparam_semantics(mech_type, 3, "netsend");
        add_nrn_artcell(mech_type, 3);
        set_pnt_receive(mech_type, net_receive_PatternStim, nullptr, num_net_receive_args());
        hoc_register_net_send_buffering(mech_type);
        hoc_register_var(hoc_scalar_double, hoc_vector_double, NULL);
    }
}
