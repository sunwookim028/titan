#ifndef MINIBATCH_PROCESS_H
#define MINIBATCH_PROCESS_H

#include "host.h"

#ifdef __cplusplus
extern "C"
{
#endif

void offloader(superbatch_data_t *data, process_data_t *proc[MAX_NUM_GPUS],\
        transfer_data_t *tran[MAX_NUM_GPUS], g3_opt_t *g3_opt,
        double *elapsed_ms);

#ifdef __cplusplus
} // end extern "C"
#endif

#endif
