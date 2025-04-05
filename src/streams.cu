#include "bwa.h"
#include "streams.cuh"
#include "CUDAKernel_memmgnt.cuh"
#include "macro.h"
#include <iostream>


/* transfer index data */
static void transferIndex(
	const bwt_t *bwt, 
	const bntseq_t *bns, 
	const uint8_t *pac,
	const kmers_bucket_t *kmerHashTab,
	process_data_t *process_instance,
    unsigned long long *allocated_size)
{
		/* CUDA GLOBAL MEMORY ALLOCATION AND TRANSFER */

	// Burrows-Wheeler Transform
		// 1. bwt_t structure
	//if (bwa_verbose>=3) fprintf(stderr, "[M::%-25s] *** bwt .......... %.2f MB\n", __func__, (float)sizeof(bwt_t)/MB_SIZE);
	bwt_t* d_bwt;
	cudaMalloc((void**)&d_bwt, sizeof(bwt_t));
	cudaMemcpy(d_bwt, bwt, sizeof(bwt_t), cudaMemcpyHostToDevice);
		// 2. int array of bwt
	//if (bwa_verbose>=3) fprintf(stderr, "[M::%-25s] *** bwt_int ...... %.2f MB\n", __func__, (float)bwt->bwt_size*sizeof(uint32_t)/MB_SIZE);
	uint32_t* d_bwt_int ;
	cudaMalloc((void**)&d_bwt_int, bwt->bwt_size*sizeof(uint32_t));
	cudaMemcpy(d_bwt_int, bwt->bwt, bwt->bwt_size*sizeof(uint32_t), cudaMemcpyHostToDevice);
		// 3. int array of Suffix Array
	//if (bwa_verbose>=3) fprintf(stderr, "[M::%-25s] *** suffix array . %.2f MB \n", __func__, (float)bwt->n_sa*sizeof(bwtint_t)/MB_SIZE);
	bwtint_t* d_bwt_sa ;
	cudaMalloc((void**)&d_bwt_sa, bwt->n_sa*sizeof(bwtint_t));
	cudaMemcpy(d_bwt_sa, bwt->sa, bwt->n_sa*sizeof(bwtint_t), cudaMemcpyHostToDevice);
		// set pointers on device's memory to bwt_int and SA
	cudaMemcpy((void**)&(d_bwt->bwt), &d_bwt_int, sizeof(uint32_t*), cudaMemcpyHostToDevice);
	cudaMemcpy((void**)&(d_bwt->sa), &d_bwt_sa, sizeof(bwtint_t*), cudaMemcpyHostToDevice);


	unsigned long long total_size = sizeof(bwt_t) +\
                                    bwt->bwt_size*sizeof(uint32_t) +\
                                    bwt->n_sa*sizeof(bwtint_t) +\
                                    bns->n_seqs*sizeof(bntann1_t) +\
                                    bns->n_holes*sizeof(bntamb1_t) +\
                                    bns->l_pac*sizeof(uint8_t);
	fprintf(stderr, "[M::%-25s] Device memory for Index ...... %.2f MB \n", __func__, (float)total_size/MB_SIZE);

	// BNS
	// First create h_bns as a copy of bns on host
	// Then allocate its member pointers on device and copy data over
	// Then copy h_bns to d_bns
	uint32_t i, size;			// loop index and length of strings
	bntseq_t* h_bns;			// host copy to modify pointers
	h_bns = (bntseq_t*)malloc(sizeof(bntseq_t));
	memcpy(h_bns, bns, sizeof(bntseq_t));
	h_bns->anns = (bntann1_t*)malloc(bns->n_seqs*sizeof(bntann1_t));
	memcpy(h_bns->ambs, bns->ambs, bns->n_holes*sizeof(bntamb1_t));
	h_bns->ambs = (bntamb1_t*)malloc(bns->n_holes*sizeof(bntamb1_t));
	memcpy(h_bns->anns, bns->anns, bns->n_seqs*sizeof(bntann1_t));

		// allocate anns.name
	for (i=0; i<bns->n_seqs; i++){
		size = strlen(bns->anns[i].name);
		// allocate this name and copy to device
		cudaMalloc((void**)&(h_bns->anns[i].name), size+1); 			// +1 for "\0"
		cudaMemcpy(h_bns->anns[i].name, bns->anns[i].name, size+1, cudaMemcpyHostToDevice);
	}
	// allocate anns.anno
	for (i=0; i<bns->n_seqs; i++){
		size = strlen(bns->anns[i].anno);
		// allocate this name and copy to device
		cudaMalloc((void**)&(h_bns->anns[i].anno), size+1); 			// +1 for "\0"
		cudaMemcpy(h_bns->anns[i].anno, bns->anns[i].anno, size+1, cudaMemcpyHostToDevice);
	}
		// now h_bns->anns has pointers of name and anno on device
		// allocate anns on device and copy data from h_bns->anns to device
	bntann1_t* temp_d_anns;
	//if (bwa_verbose>=3) fprintf(stderr, "[M::%-25s] *** bns.anns ..... %.2f MB\n", __func__, (float)bns->n_seqs*sizeof(bntann1_t)/MB_SIZE);
	cudaMalloc((void**)&temp_d_anns, bns->n_seqs*sizeof(bntann1_t));
	cudaMemcpy(temp_d_anns, h_bns->anns, bns->n_seqs*sizeof(bntann1_t), cudaMemcpyHostToDevice);
		// now assign this pointer to h_bns->anns
	h_bns->anns = temp_d_anns;

		// allocate bns->ambs on device and copy data to device
	//if (bwa_verbose>=3) fprintf(stderr, "[M::%-25s] *** bns.ambs ..... %.2f MB\n", __func__, (float)bns->n_holes*sizeof(bntamb1_t)/MB_SIZE);
	cudaMalloc((void**)&h_bns->ambs, bns->n_holes*sizeof(bntamb1_t));
	cudaMemcpy(h_bns->ambs, bns->ambs, bns->n_holes*sizeof(bntamb1_t), cudaMemcpyHostToDevice);

		// finally allocate d_bns and copy from h_bns
	//if (bwa_verbose>=3) fprintf(stderr, "[M::%-25s] *** bns .......... %.2f MB\n", __func__, (float)sizeof(bntseq_t)/MB_SIZE);
	bntseq_t* d_bns;
	cudaMalloc((void**)&d_bns, sizeof(bntseq_t));
	cudaMemcpy(d_bns, h_bns, sizeof(bntseq_t), cudaMemcpyHostToDevice);

	// PAC
	//if (bwa_verbose>=3) fprintf(stderr, "[M::%-25s] *** pac .......... %.2f MB\n", __func__, (float)bns->l_pac*sizeof(uint8_t)/MB_SIZE);
	uint8_t* d_pac ;
	cudaMalloc((void**)&d_pac, bns->l_pac/4*sizeof(uint8_t)); 		// l_pac is length of ref seq
	cudaMemcpy(d_pac, pac, bns->l_pac/4*sizeof(uint8_t), cudaMemcpyHostToDevice); 		// divide by 4 because 2-bit encoding

	// K-MER HASH TABLE
	//if (bwa_verbose>=3) fprintf(stderr, "[M::%-25s] *** kmer ......... %.2f MB\n", __func__, (float)pow4(KMER_K)*sizeof(kmers_bucket_t)/MB_SIZE);
	kmers_bucket_t* d_kmerHashTab ;
	cudaMalloc((void**)&d_kmerHashTab, pow4(KMER_K)*sizeof(kmers_bucket_t)); 		// l_pac is length of ref seq
	cudaMemcpy(d_kmerHashTab, kmerHashTab, pow4(KMER_K)*sizeof(kmers_bucket_t), cudaMemcpyHostToDevice); 		// divide by 4 because 2-bit encoding


	// output
	process_instance->d_bwt = d_bwt;
	process_instance->d_bns = d_bns;
	process_instance->d_pac = d_pac;
	process_instance->d_kmerHashTab = d_kmerHashTab;
}

/* transfer user-defined optinos */
static void transferOptions(
	const mem_opt_t *opt, 
	mem_pestat_t *pes0,
	process_data_t *process_instance,
    unsigned long long *allocated_size)
{
	// matching and mapping options (opt)
	mem_opt_t* d_opt;
	cudaMalloc((void**)&d_opt, sizeof(mem_opt_t));
	cudaMemcpy(d_opt, opt, sizeof(mem_opt_t), cudaMemcpyHostToDevice);

	// paired-end stats: only allocate on device
	mem_pestat_t* d_pes;
	if (opt->flag&MEM_F_PE){
		fprintf(stderr, "[M::%-25s] pestat ....... %.2f MB\n", __func__, (float)4*sizeof(mem_pestat_t)/MB_SIZE);
		cudaMalloc((void**)&d_pes, 4*sizeof(mem_pestat_t));
	}

	// output
	process_instance->d_opt = d_opt;
	process_instance->d_pes = d_pes;
	process_instance->h_pes0 = pes0;
}

inline int allocateAuxIntervals(process_data_t *process_instance, g3_opt_t *g3_opt)
{
    smem_aux_t *h_aux;
    smem_aux_t *d_aux; 
    h_aux = (smem_aux_t*)malloc(sizeof(smem_aux_t) * g3_opt->batch_size);
    if(h_aux == nullptr)
    {
        fprintf(stderr, "malloc error at [%s], h_aux\n", __func__);
        exit(1);
    }
    CUDA_CHECK(cudaMalloc((void**)&d_aux, sizeof(smem_aux_t) * g3_opt->batch_size));

    CUDA_CHECK(cudaMemcpy(d_aux, h_aux, sizeof(smem_aux_t) * g3_opt->batch_size, cudaMemcpyHostToDevice));
    process_instance->d_aux = d_aux;
    free(h_aux);
    return sizeof(bwtintv_t) * g3_opt->batch_size; 
}


/* allocate memory for intermediate data on GPU
	send pointer to process_instance
 */
void allocateIntermediateData(process_data_t *process_instance, g3_opt_t *g3_opt, unsigned long long *allocated_size){
    //auto old_value = bwa_verbose;
    //bwa_verbose = 3;
	unsigned long long total_size = (g3_opt->batch_size) *sizeof(smem_aux_t) + (g3_opt->batch_size) *sizeof(mem_seed_v) + (g3_opt->batch_size) *sizeof(mem_chain_v) + (g3_opt->batch_size) *500*sizeof(seed_record_t) + (g3_opt->batch_size) *sizeof(mem_alnreg_v) + (g3_opt->batch_size) *sizeof(mem_aln_v) + 4*5*(g3_opt->batch_size) *sizeof(int);
	fprintf(stderr, "[M::%-25s] total intermediate data ..... %.2f MB\n", __func__, (float)total_size/MB_SIZE);
	fprintf(stderr, "[M::%-25s] pre-allocating device memory to align batch of%ld reads...\n", __func__, (g3_opt->batch_size));

    auto size1 = allocateAuxIntervals(process_instance, g3_opt);
	//if (bwa_verbose>=3) fprintf(stderr, "[M::%-25s] *** aux intervals ..... %.2f MB\n", __func__, size1/GB_SIZE); // = (24,916 + 378) * MBSIZE (B)
	//CUDA_CHECK(cudaMalloc((void**)&(process_instance->d_aux), (g3_opt->batch_size) *sizeof(smem_aux_t)));

	//if (bwa_verbose>=3) fprintf(stderr, "[M::%-25s] *** seeds array  ...... %ld MB\n", __func__, (g3_opt->batch_size) *sizeof(mem_seed_v)/MB_SIZE);
	CUDA_CHECK(cudaMalloc((void**)&(process_instance->d_seq_seeds), (g3_opt->batch_size) *sizeof(mem_seed_v)));

	//if (bwa_verbose>=3) fprintf(stderr, "[M::%-25s] *** chains ............ %ld MB\n", __func__, (g3_opt->batch_size) *sizeof(mem_chain_v)/MB_SIZE);
	CUDA_CHECK(cudaMalloc((void**)&(process_instance->d_chains), (g3_opt->batch_size) *sizeof(mem_chain_v)));

	//if (bwa_verbose>=3) fprintf(stderr, "[M::%-25s] *** seed records ...... %ld MB\n", __func__, (g3_opt->batch_size) *500*sizeof(seed_record_t)/MB_SIZE);
	CUDA_CHECK(cudaMalloc((void**)&(process_instance->d_seed_records), (g3_opt->batch_size) *MAX_NUM_SW_SEEDS*sizeof(seed_record_t)));	// allocate enough for all seeds

	CUDA_CHECK(cudaMalloc((void**)&(process_instance->d_Nseeds), sizeof(int)));

	//if (bwa_verbose>=3) fprintf(stderr, "[M::%-25s] *** alignment regs .... %ld MB\n", __func__, (g3_opt->batch_size) *sizeof(mem_alnreg_v)/MB_SIZE);
	CUDA_CHECK(cudaMalloc((void**)&(process_instance->d_regs), (g3_opt->batch_size) *sizeof(mem_alnreg_v)));

	//if (bwa_verbose>=3) fprintf(stderr, "[M::%-25s] *** alignments ...... %ld MB\n", __func__, (g3_opt->batch_size) *sizeof(mem_aln_v)/MB_SIZE);
	CUDA_CHECK(cudaMalloc((void**)&(process_instance->d_alns), (g3_opt->batch_size) *sizeof(mem_aln_v)));

	//if (bwa_verbose>=3) fprintf(stderr, "[M::%-25s] *** sorting keys .... %ld MB\n", __func__, 4*5*(g3_opt->batch_size) *sizeof(int)/MB_SIZE);
	CUDA_CHECK(cudaMalloc((void**)&process_instance->d_sortkeys_in, (g3_opt->batch_size) *5*sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**)&process_instance->d_sortkeys_out, (g3_opt->batch_size) *5*sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**)&process_instance->d_seqIDs_in, (g3_opt->batch_size) *5*sizeof(int)));
	CUDA_CHECK(cudaMalloc((void**)&process_instance->d_seqIDs_out, (g3_opt->batch_size) *5*sizeof(int)));
    //bwa_verbose = old_value;
}

/* transfer index data */
static void transferFmIndex(
        process_data_t *process_instance,
        const fmIndex *idx,
    unsigned long long *allocated_size)
{
    /**
     * Reference data to transfer:
     *      count, count2
     *      cpOcc, cpOcc2
     *      oneHot, sentinelIndex, firstBase
     */
    fmIndex hostFmIndex;
    cudaError_t err;

    // FOR BWT-2
    uint64_t *d_one_hot;
    int sizeOneHot = 64 * sizeof(uint64_t);
    fprintf(stderr, "[%s] *** one_hot ......... %.2f MB\n", __func__, (float)sizeOneHot/MB_SIZE);
    cudaMalloc((void**)&d_one_hot, sizeOneHot);
    cudaMemcpy(d_one_hot, idx->oneHot, sizeOneHot, cudaMemcpyHostToDevice);

    CP_OCC *d_cp_occ;
    int64_t cp_occ_size = idx->cpOccSize;
    fprintf(stderr, "[%s] *** cp_occ ......... %.2f MB\n", __func__, (float)cp_occ_size*sizeof(CP_OCC)/MB_SIZE);
    cudaMalloc((void**)&d_cp_occ, cp_occ_size*sizeof(CP_OCC));
    cudaMemcpy(d_cp_occ, idx->cpOcc, cp_occ_size*sizeof(CP_OCC), cudaMemcpyHostToDevice);

    int64_t *d_count;
    int sizeCount = sizeof(int64_t) * 5;
    fprintf(stderr, "[%s] *** count ......... %.2f MB\n", __func__, (float)sizeCount/MB_SIZE);
    cudaMalloc((void**)&d_count, sizeCount);
    cudaMemcpy(d_count, idx->count, sizeCount, cudaMemcpyHostToDevice);


    CP_OCC2 *d_cp_occ2;
    fprintf(stderr, "[%s] *** cp_occ2 ......... %.2f MB\n", __func__, (float)cp_occ_size*sizeof(CP_OCC2)/MB_SIZE);
    cudaMalloc((void**)&d_cp_occ2, cp_occ_size*sizeof(CP_OCC2));
    cudaMemcpy(d_cp_occ2, idx->cpOcc2, cp_occ_size*sizeof(CP_OCC2), cudaMemcpyHostToDevice);

    int64_t *d_count2;
    int sizeCount2 = sizeof(int64_t) * 17;
    fprintf(stderr, "[%s] *** count2......... %.2f MB\n", __func__, (float)sizeCount2/MB_SIZE);
    cudaMalloc((void**)&d_count2, sizeCount2);
    cudaMemcpy(d_count2, idx->count2, sizeCount2, cudaMemcpyHostToDevice);

    uint8_t *d_first_base;
    cudaMalloc((void**)&d_first_base, sizeof(uint8_t));
    cudaMemcpy(d_first_base, idx->firstBase, sizeof(uint8_t), cudaMemcpyHostToDevice);

    int64_t *deviceSentinelIndex;
    cudaMalloc((void**)&deviceSentinelIndex, sizeof(int64_t));
    cudaMemcpy(deviceSentinelIndex, idx->sentinelIndex, sizeof(int64_t), cudaMemcpyHostToDevice);

    fprintf(stderr, "*******\n");
fprintf(stderr, "transferred sentinel index = %ld to device\n", *(idx->sentinelIndex));
    fprintf(stderr, "*******\n");

    hostFmIndex.oneHot = d_one_hot;
    hostFmIndex.cpOcc = d_cp_occ;
    hostFmIndex.cpOcc2 = d_cp_occ2;
    hostFmIndex.count = d_count;
    hostFmIndex.count2 = d_count2;
    hostFmIndex.firstBase = d_first_base;
    hostFmIndex.sentinelIndex = deviceSentinelIndex;

    fmIndex *deviceFmIndex;
    cudaMalloc((void**)&deviceFmIndex, sizeof(fmIndex));
    cudaMemcpy(deviceFmIndex, &hostFmIndex, sizeof(fmIndex), cudaMemcpyHostToDevice);

    // output
    process_instance->d_fmIndex = deviceFmIndex;
}

void newProcess(
        process_data_t ** output,
        int gpuid,
	const mem_opt_t *opt, 
	mem_pestat_t *pes0,
	const bwt_t *bwt, 
	const bntseq_t *bns, 
	const uint8_t *pac,
	const kmers_bucket_t *kmerHashTab,
    const fmIndex *hostFmIndex,
    g3_opt_t *g3_opt
)
{
    if(cudaSetDevice(gpuid) != cudaSuccess){
        std::cerr << "newProcess: cudaSetDevice failed" << std::endl;
        exit(1);
    }
    int current;
    cudaGetDevice(&current);
    if(current != gpuid){
         std::cerr << "GPU " << gpuid << "  " << "newProcess: cudaSetDevice is wrong" << std::endl;
        return;
    }
    // new instance in memory
    process_data_t *instance = (process_data_t*)calloc(1, sizeof(process_data_t));
    instance->gpu_no = gpuid;

    unsigned long long allocated_size;

	// user-defined options
	transferOptions(opt, pes0, instance, &allocated_size);
    
	// transfer index data
	transferIndex(bwt, bns, pac, kmerHashTab, instance, &allocated_size);

    transferFmIndex(instance, hostFmIndex, &allocated_size);

	// init memory management
	instance->d_buffer_pools = CUDA_BufferInit(g3_opt->batch_size);

	// initialize intermediate processing memory on device
	allocateIntermediateData(instance, g3_opt, &allocated_size);

	// initialize pinned memory for reads on host
	CUDA_CHECK(cudaMallocHost((void**)&instance->h_seqs, (g3_opt->batch_size)*sizeof(bseq1_t)));
	CUDA_CHECK(cudaMallocHost((void**)&instance->h_seq_name_ptr, (g3_opt->batch_size) * 1 ));
	CUDA_CHECK(cudaMallocHost((void**)&instance->h_seq_comment_ptr, (g3_opt->batch_size) * 1));
	CUDA_CHECK(cudaMallocHost((void**)&instance->h_seq_seq_ptr, (g3_opt->batch_size) * SEQ_MAXLEN ));
	CUDA_CHECK(cudaMallocHost((void**)&instance->h_seq_qual_ptr, (g3_opt->batch_size) * SEQ_MAXLEN ));
	CUDA_CHECK(cudaMallocHost((void**)&instance->h_seq_sam_ptr, (g3_opt->batch_size) ));
	
	if (instance->h_seqs == nullptr || instance->h_seq_name_ptr == nullptr ||
		instance->h_seq_comment_ptr == nullptr || instance->h_seq_seq_ptr == nullptr ||
		instance->h_seq_qual_ptr == nullptr || instance->h_seq_sam_ptr == nullptr)
	{
        fprintf(stderr, "[M::%-25s] can't malloc minibatch on host\n", __func__);
        exit(1);
    }

	// initialize memory for reads on device
	unsigned long long total_size = g3_opt->batch_size*(sizeof(bseq1_t) + 1 + 1 + SEQ_MAXLEN + SEQ_MAXLEN + 1);
	fprintf(stderr, "[M::%-25s] d_seqs (transf) .. %llu MB\n", __func__, total_size/MB_SIZE);
	CUDA_CHECK(cudaMalloc((void**)&instance->d_seqs, g3_opt->batch_size*sizeof(bseq1_t)));
	CUDA_CHECK(cudaMalloc((void**)&instance->d_seq_name_ptr, g3_opt->batch_size * 1));
	CUDA_CHECK(cudaMalloc((void**)&instance->d_seq_comment_ptr, g3_opt->batch_size * 1));
	CUDA_CHECK(cudaMalloc((void**)&instance->d_seq_seq_ptr, g3_opt->batch_size * SEQ_MAXLEN));
	CUDA_CHECK(cudaMalloc((void**)&instance->d_seq_qual_ptr, g3_opt->batch_size * SEQ_MAXLEN));
	CUDA_CHECK(cudaMalloc((void**)&instance->d_seq_sam_ptr, g3_opt->batch_size * 1));
	CUDA_CHECK(cudaMalloc((void**)&instance->d_seq_sam_size, sizeof(int)));

	if (instance->d_seqs == nullptr || instance->d_seq_name_ptr == nullptr ||
		instance->d_seq_comment_ptr == nullptr || instance->d_seq_seq_ptr == nullptr ||
		instance->d_seq_qual_ptr == nullptr || instance->d_seq_sam_ptr == nullptr)
	{
        fprintf(stderr, "[M::%-25s] can't malloc minibatch on GPU\n", __func__);
        exit(1);
    }

	// initialize a cuda stream for processing
	instance->CUDA_stream = malloc(sizeof(cudaStream_t));
	CUDA_CHECK(cudaStreamCreate((cudaStream_t*)instance->CUDA_stream));

    *output = instance;
}


void newTransfer(transfer_data_t **output, int gpuid, g3_opt_t *g3_opt){
    if(cudaSetDevice(gpuid) != cudaSuccess){
        std::cerr << "newTransfer: cudaSetDevice failed" << std::endl;
        exit(1);
    }

    int current;
    cudaGetDevice(&current);
    if(current != gpuid){
         std::cerr << "GPU " << gpuid << "  " << "newTransfer: cudaSetDevice is wrong" << std::endl;
        return;
    }
    transfer_data_t *instance = (transfer_data_t*)calloc(1, sizeof(transfer_data_t));
    instance->gpu_no = gpuid;

	// initialize pinned memory for reads on host
	CUDA_CHECK(cudaMallocHost((void**)&instance->h_seqs, (g3_opt->batch_size)*sizeof(bseq1_t)));
	CUDA_CHECK(cudaMallocHost((void**)&instance->h_seq_name_ptr, (g3_opt->batch_size) * 1 ));
	CUDA_CHECK(cudaMallocHost((void**)&instance->h_seq_comment_ptr, (g3_opt->batch_size) * 1));
	CUDA_CHECK(cudaMallocHost((void**)&instance->h_seq_seq_ptr, (g3_opt->batch_size) * SEQ_MAXLEN ));
	CUDA_CHECK(cudaMallocHost((void**)&instance->h_seq_qual_ptr, (g3_opt->batch_size) * SEQ_MAXLEN ));
	CUDA_CHECK(cudaMallocHost((void**)&instance->h_seq_sam_ptr, (g3_opt->batch_size) ));
	
	if (instance->h_seqs == nullptr || instance->h_seq_name_ptr == nullptr ||
		instance->h_seq_comment_ptr == nullptr || instance->h_seq_seq_ptr == nullptr ||
		instance->h_seq_qual_ptr == nullptr || instance->h_seq_sam_ptr == nullptr)
	{
        fprintf(stderr, "[M::%-25s] can't malloc minibatch on host\n", __func__);
        exit(1);
    }

	// initialize memory for reads on device
	unsigned long long total_size = g3_opt->batch_size*(sizeof(bseq1_t) + 1 + 1 + SEQ_MAXLEN + SEQ_MAXLEN + 1);
	fprintf(stderr, "[M::%-25s] d_seqs (transf) .. %llu MB\n", __func__, total_size/MB_SIZE);
	CUDA_CHECK(cudaMalloc((void**)&instance->d_seqs, g3_opt->batch_size*sizeof(bseq1_t)));
	CUDA_CHECK(cudaMalloc((void**)&instance->d_seq_name_ptr, g3_opt->batch_size * 1));
	CUDA_CHECK(cudaMalloc((void**)&instance->d_seq_comment_ptr, g3_opt->batch_size * 1));
	CUDA_CHECK(cudaMalloc((void**)&instance->d_seq_seq_ptr, g3_opt->batch_size * SEQ_MAXLEN));
	CUDA_CHECK(cudaMalloc((void**)&instance->d_seq_qual_ptr, g3_opt->batch_size * SEQ_MAXLEN));
	CUDA_CHECK(cudaMalloc((void**)&instance->d_seq_sam_ptr, g3_opt->batch_size * 1));
	CUDA_CHECK(cudaMalloc((void**)&instance->d_seq_sam_size, sizeof(int)));

	if (instance->d_seqs == nullptr || instance->d_seq_name_ptr == nullptr ||
		instance->d_seq_comment_ptr == nullptr || instance->d_seq_seq_ptr == nullptr ||
		instance->d_seq_qual_ptr == nullptr || instance->d_seq_sam_ptr == nullptr)
	{
        fprintf(stderr, "[M::%-25s] can't malloc minibatch on GPU\n", __func__);
        exit(1);
    }

	// initialize a cuda stream for transfer
	instance->CUDA_stream = malloc(sizeof(cudaStream_t));
	CUDA_CHECK(cudaStreamCreate((cudaStream_t*)instance->CUDA_stream));

    *output = instance;
}



void swapData(process_data_t *proc, transfer_data_t *tran){
#define P2T(field) tran->field = proc->field
#define T2P(field) proc->field = tran->field
#define SWAP(field)\
    {\
        auto tmp = proc->field;\
        proc->field = tran->field;\
        tran->field = tmp;\
    }
	// swap host pointers
	{ auto tmp = proc->h_seqs; proc->h_seqs = tran->h_seqs; tran->h_seqs = tmp; }
	{ auto tmp = proc->h_seq_name_ptr; proc->h_seq_name_ptr = tran->h_seq_name_ptr; tran->h_seq_name_ptr = tmp; }
	{ auto tmp = proc->h_seq_comment_ptr; proc->h_seq_comment_ptr = tran->h_seq_comment_ptr; tran->h_seq_comment_ptr = tmp; }
	{ auto tmp = proc->h_seq_seq_ptr; proc->h_seq_seq_ptr = tran->h_seq_seq_ptr; tran->h_seq_seq_ptr = tmp; }
	{ auto tmp = proc->h_seq_qual_ptr; proc->h_seq_qual_ptr = tran->h_seq_qual_ptr; tran->h_seq_qual_ptr = tmp; }
	{ auto tmp = proc->h_seq_sam_ptr; proc->h_seq_sam_ptr = tran->h_seq_sam_ptr; tran->h_seq_sam_ptr = tmp; }
	// swap device pointers
	{ auto tmp = proc->d_seqs; proc->d_seqs = tran->d_seqs; tran->d_seqs = tmp; }
	{ auto tmp = proc->d_seq_name_ptr; proc->d_seq_name_ptr = tran->d_seq_name_ptr; tran->d_seq_name_ptr = tmp; }
	{ auto tmp = proc->d_seq_comment_ptr; proc->d_seq_comment_ptr = tran->d_seq_comment_ptr; tran->d_seq_comment_ptr = tmp; }
	{ auto tmp = proc->d_seq_seq_ptr; proc->d_seq_seq_ptr = tran->d_seq_seq_ptr; tran->d_seq_seq_ptr = tmp; }
	{ auto tmp = proc->d_seq_qual_ptr; proc->d_seq_qual_ptr = tran->d_seq_qual_ptr; tran->d_seq_qual_ptr = tmp; }
	{ auto tmp = proc->d_seq_sam_ptr; proc->d_seq_sam_ptr = tran->d_seq_sam_ptr; tran->d_seq_sam_ptr = tmp; }

    // FIXME keep only the necessary functionalities.

    // Batch size & offset.
	{ 
        SWAP(batch_size);
        SWAP(batch_offset);
    }

    // Alignment results.
    {
        P2T(d_total_alns_num);
        P2T(d_alns_num);
        P2T(d_alns_offset);
        P2T(d_alns_rid);
        P2T(d_alns_pos);
        P2T(d_total_cigar_len);
        P2T(d_alns_cigar_len);
        P2T(d_alns_cigar_offset);
        P2T(d_alns_cigar);
    }

    return;
}

void CUDATransferSeqsIn(transfer_data_t *transfer_data){
    int gpuid = transfer_data->gpu_no;
    if(cudaSetDevice(transfer_data->gpu_no) != cudaSuccess){
        std::cerr << "CUDATransferSeqsIn: cudaSetDevice failed" << std::endl;
        exit(1);
    }
    int current;
    cudaGetDevice(&current);
    if(current != gpuid){
         std::cerr << "GPU " << gpuid << "  " << "CUDATransferSeqsIn: cudaSetDevice is wrong" << std::endl;
        return;
    }
	cudaStream_t *transfer_stream = (cudaStream_t*)(transfer_data->CUDA_stream);
	// copy seqs to device
	CUDA_CHECK(cudaMemcpyAsync(transfer_data->d_seqs, transfer_data->h_seqs, transfer_data->batch_size*sizeof(bseq1_t), cudaMemcpyHostToDevice, *transfer_stream));
	// copy name, seq, comment, qual to device
	CUDA_CHECK(cudaMemcpyAsync(transfer_data->d_seq_name_ptr, transfer_data->h_seq_name_ptr, transfer_data->h_seq_name_size, cudaMemcpyHostToDevice, *transfer_stream));
	CUDA_CHECK(cudaMemcpyAsync(transfer_data->d_seq_seq_ptr, transfer_data->h_seq_seq_ptr, transfer_data->h_seq_seq_size, cudaMemcpyHostToDevice, *transfer_stream));
	CUDA_CHECK(cudaMemcpyAsync(transfer_data->d_seq_comment_ptr, transfer_data->h_seq_comment_ptr, transfer_data->h_seq_comment_size, cudaMemcpyHostToDevice, *transfer_stream));
	CUDA_CHECK(cudaMemcpyAsync(transfer_data->d_seq_qual_ptr, transfer_data->h_seq_qual_ptr, transfer_data->h_seq_qual_size, cudaMemcpyHostToDevice, *transfer_stream));

	CUDA_CHECK(cudaStreamSynchronize(*transfer_stream));
}

/* copy sam output to host */
void CUDATransferSamOut(transfer_data_t *transfer_data){
    int gpuid = transfer_data->gpu_no;
    if(cudaSetDevice(transfer_data->gpu_no) != cudaSuccess){
        std::cerr << "CUDATransferSamOut: cudaSetDevice failed" << std::endl;
        exit(1);
    }
    int current;
    cudaGetDevice(&current);
    if(current != gpuid){
         std::cerr << "GPU " << gpuid << "  " << "CUDATransferSamOut: cudaSetDevice is wrong" << std::endl;
        return;
    }
	cudaStream_t *transfer_stream = (cudaStream_t*)(transfer_data->CUDA_stream);
	CUDA_CHECK(cudaMemcpyAsync(transfer_data->h_seqs, transfer_data->d_seqs, transfer_data->batch_size*sizeof(bseq1_t), cudaMemcpyDeviceToHost, *transfer_stream));
	// transfer all SAM from device to host
	// first find the total size of all SAM's
	int sam_size;
	CUDA_CHECK(cudaMemcpyAsync(&sam_size, transfer_data->d_seq_sam_size, sizeof(int), cudaMemcpyDeviceToHost, *transfer_stream));
	// now copy
	CUDA_CHECK(cudaMemcpyAsync(transfer_data->h_seq_sam_ptr, transfer_data->d_seq_sam_ptr, sam_size, cudaMemcpyDeviceToHost, *transfer_stream));

	cudaStreamSynchronize(*transfer_stream);

	// after GPU processing, seqs[i].sam are offset. Now we need to convert this offset to actual location
	bseq1_t *seqs = transfer_data->h_seqs;
	char *sam = transfer_data->h_seq_sam_ptr;
	for (int i=0; i<transfer_data->batch_size; i++)
		seqs[i].sam = sam + (long)seqs[i].sam;
}



void resetTransfer(transfer_data_t *transfer_data){
	// reset name, seq, comment, qual sizes
	transfer_data->h_seq_name_size = 0;
	transfer_data->h_seq_seq_size = 0;
	transfer_data->h_seq_comment_size = 0;
	transfer_data->h_seq_qual_size = 0;
	// reset batch_size
	transfer_data->batch_size = 0;
}
