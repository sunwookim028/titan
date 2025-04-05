/* The MIT License

   Copyright (c) 2018-     Dana-Farber Cancer Institute
                 2009-2018 Broad Institute, Inc.
                 2008-2009 Genome Research Ltd. (GRL)

   Permission is hereby granted, free of charge, to any person obtaining
   a copy of this software and associated documentation files (the
   "Software"), to deal in the Software without restriction, including
   without limitation the rights to use, copy, modify, merge, publish,
   distribute, sublicense, and/or sell copies of the Software, and to
   permit persons to whom the Software is furnished to do so, subject to
   the following conditions:

   The above copyright notice and this permission notice shall be
   included in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
   NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
   BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
   ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
   CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.
*/
#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <cfloat>
#include <ctype.h>
#include <math.h>
#include "macro.h"
#include "FMI_wrapper.h"
#include <time.h>

#include <iostream>
#include <iomanip>

#include "fastmap.h"
#include "utils.h"

#include "khash.h"
KHASH_MAP_INIT_STR(str, int)


float tprof[MAX_NUM_GPUS][MAX_NUM_STEPS];
extern unsigned char nst_nt4_table[256];
	
extern void pipeline(ktp_aux_t *aux, g3_opt_t *g3_opt);

g3_opt_t *g3_opt_init()
{
    g3_opt_t *o;
    o = (g3_opt_t*)calloc(1, sizeof(g3_opt_t));
    o->num_use_gpus = 1;
    o->baseline = 0;
    o->verbosity = 0;
    o->print_mask = 0;
    o->batch_size = 80000;
    return o;
}

mem_opt_t *mem_opt_init()
{
	mem_opt_t *o;
	o = (mem_opt_t*)calloc(1, sizeof(mem_opt_t));
	o->flag = 0;
	o->a = 1; o->b = 4;
	o->o_del = o->o_ins = 6;
	o->e_del = o->e_ins = 1;
	o->w = 100;
	o->T = 30;
	o->zdrop = 100;
	o->pen_unpaired = 17;
	o->pen_clip5 = o->pen_clip3 = 5;

	o->max_mem_intv = 20;

	o->min_seed_len = 19;
	o->split_width = 10;
	o->max_occ = 500;
	o->max_chain_gap = 10000;
	o->max_ins = 10000;
	o->mask_level = 0.50;
	o->drop_ratio = 0.50;
	o->XA_drop_ratio = 0.80;
	o->split_factor = 1.5;
	o->chunk_size = 30000000;
	o->n_threads = 1;
	o->max_XA_hits = 5;
	o->max_XA_hits_alt = 200;
	o->max_matesw = 50;
	o->mask_level_redun = 0.95;
	o->min_chain_weight = 0;
	o->max_chain_extend = 1<<30;
	o->mapQ_coef_len = 50; o->mapQ_coef_fac = log(o->mapQ_coef_len);
	bwa_fill_scmat(o->a, o->b, o->mat);
	return o;
}

void bwa_fill_scmat(int a, int b, int8_t mat[25])
{
	int i, j, k;
	for (i = k = 0; i < 4; ++i) {
		for (j = 0; j < 4; ++j)
			mat[k++] = i == j? a : -b;
		mat[k++] = -1; // ambiguous base
	}
	for (j = 0; j < 5; ++j) mat[k++] = -1;
}

static void update_a(mem_opt_t *opt, const mem_opt_t *opt0)
{
	if (opt0->a) { // matching score is changed
		if (!opt0->b) opt->b *= opt->a;
		if (!opt0->T) opt->T *= opt->a;
		if (!opt0->o_del) opt->o_del *= opt->a;
		if (!opt0->e_del) opt->e_del *= opt->a;
		if (!opt0->o_ins) opt->o_ins *= opt->a;
		if (!opt0->e_ins) opt->e_ins *= opt->a;
		if (!opt0->zdrop) opt->zdrop *= opt->a;
		if (!opt0->pen_clip5) opt->pen_clip5 *= opt->a;
		if (!opt0->pen_clip3) opt->pen_clip3 *= opt->a;
		if (!opt0->pen_unpaired) opt->pen_unpaired *= opt->a;
	}
}


/*********************
 * Full index reader *
 *********************/
bntseq_t *bns_restore_core(const char *ann_filename, const char* amb_filename, const char* pac_filename)
{
	char str[8192];
	FILE *fp;
	const char *fname;
	bntseq_t *bns;
	long long xx;
	int i;
	int scanres;
	bns = (bntseq_t*)calloc(1, sizeof(bntseq_t));
	{ // read .ann
		fp = xopen(fname = ann_filename, "r");
		scanres = fscanf(fp, "%lld%d%u", &xx, &bns->n_seqs, &bns->seed);
		if (scanres != 3) goto badread;
		bns->l_pac = xx;
		bns->anns = (bntann1_t*)calloc(bns->n_seqs, sizeof(bntann1_t));
		for (i = 0; i < bns->n_seqs; ++i) {
			bntann1_t *p = bns->anns + i;
			char *q = str;
			int c;
			// read gi and sequence name
			scanres = fscanf(fp, "%u%s", &p->gi, str);
			if (scanres != 2) goto badread;
			p->name = strdup(str);
			// read fasta comments 
			while (q - str < sizeof(str) - 1 && (c = fgetc(fp)) != '\n' && c != EOF) *q++ = c;
			while (c != '\n' && c != EOF) c = fgetc(fp);
			if (c == EOF) {
				scanres = EOF;
				goto badread;
			}
			*q = 0;
			if (q - str > 1 && strcmp(str, " (null)") != 0) p->anno = strdup(str + 1); // skip leading space
			else p->anno = strdup("");
			// read the rest
			scanres = fscanf(fp, "%lld%d%d", &xx, &p->len, &p->n_ambs);
			if (scanres != 3) goto badread;
			p->offset = xx;
		}
		err_fclose(fp);
	}
	{ // read .amb
		int64_t l_pac;
		int32_t n_seqs;
		fp = xopen(fname = amb_filename, "r");
		scanres = fscanf(fp, "%lld%d%d", &xx, &n_seqs, &bns->n_holes);
		if (scanres != 3) goto badread;
		l_pac = xx;
		xassert(l_pac == bns->l_pac && n_seqs == bns->n_seqs, "inconsistent .ann and .amb files.");
		bns->ambs = bns->n_holes? (bntamb1_t*)calloc(bns->n_holes, sizeof(bntamb1_t)) : 0;
		for (i = 0; i < bns->n_holes; ++i) {
			bntamb1_t *p = bns->ambs + i;
			scanres = fscanf(fp, "%lld%d%s", &xx, &p->len, str);
			if (scanres != 3) goto badread;
			p->offset = xx;
			p->amb = str[0];
		}
		err_fclose(fp);
	}
	{ // open .pac
		bns->fp_pac = xopen(pac_filename, "rb");
	}
	return bns;

 badread:
	if (EOF == scanres) {
		err_fatal(__func__, "Error reading %s : %s\n", fname, ferror(fp) ? strerror(errno) : "Unexpected end of file");
	}
	err_fatal(__func__, "Parse error reading %s\n", fname);
}

bntseq_t *bns_restore(const char *prefix)
{  
	char ann_filename[1024], amb_filename[1024], pac_filename[1024], alt_filename[1024];
	FILE *fp;
	bntseq_t *bns;
	strcat(strcpy(ann_filename, prefix), ".ann");
	strcat(strcpy(amb_filename, prefix), ".amb");
	strcat(strcpy(pac_filename, prefix), ".pac");
	bns = bns_restore_core(ann_filename, amb_filename, pac_filename);
	if (bns == 0) return 0;
	if ((fp = fopen(strcat(strcpy(alt_filename, prefix), ".alt"), "r")) != 0) { // read .alt file if present
		char str[1024];
		khash_t(str) *h;
		int c, i, absent;
		khint_t k;
		h = kh_init(str);
		for (i = 0; i < bns->n_seqs; ++i) {
			k = kh_put(str, h, bns->anns[i].name, &absent);
			kh_val(h, k) = i;
		}
		i = 0;
		while ((c = fgetc(fp)) != EOF) {
			if (c == '\t' || c == '\n' || c == '\r') {
				str[i] = 0;
				if (str[0] != '@') {
					k = kh_get(str, h, str);
					if (k != kh_end(h))
						bns->anns[kh_val(h, k)].is_alt = 1;
				}
				while (c != '\n' && c != EOF) c = fgetc(fp);
				i = 0;
			} else {
				if (i >= 1022) {
					fprintf(stderr, "[E::%s] sequence name longer than 1023 characters. Abort!\n", __func__);
					exit(1);
				}
				str[i++] = c;
			}
		}
		kh_destroy(str, h);
		fclose(fp);
	}
	return bns;
}

void bns_destroy(bntseq_t *bns)
{
	if (bns == 0) return;
	else {
		int i;
		if (bns->fp_pac) err_fclose(bns->fp_pac);
		free(bns->ambs);
		for (i = 0; i < bns->n_seqs; ++i) {
			free(bns->anns[i].name);
			free(bns->anns[i].anno);
		}
		free(bns->anns);
		free(bns);
	}
}

char *bwa_idx_infer_prefix(const char *hint)
{
	char *prefix;
	int l_hint;
	FILE *fp;
	l_hint = strlen(hint);
	prefix = (char*)malloc(l_hint + 3 + 4 + 1);
	strcpy(prefix, hint);
    //fprintf(stderr, "prefix = %s\n", prefix);
	strcpy(prefix + l_hint, ".64.bwt");
	if ((fp = fopen(prefix, "rb")) != 0) {
		fclose(fp);
		prefix[l_hint + 3] = 0;
		return prefix;
	} else {
		strcpy(prefix + l_hint, ".bwt");
        //fprintf(stderr, "prefix2 %s\n", prefix);
		if ((fp = fopen(prefix, "rb")) == 0) {
			free(prefix);
			return 0;
		} else {
			fclose(fp);
			prefix[l_hint] = 0;
			return prefix;
		}
	}
}

bwt_t *bwa_idx_load_bwt(const char *hint)
{
	char *tmp, *prefix;
	bwt_t *bwt;
	prefix = bwa_idx_infer_prefix(hint);
	if (prefix == 0) {
		fprintf(stderr, "[E::%s] fail to locate the index files\n", __func__);
		return 0;
	}
	tmp = (char*)calloc(strlen(prefix) + 5, 1);
	strcat(strcpy(tmp, prefix), ".bwt"); // FM-index
	bwt = bwt_restore_bwt(tmp);
	strcat(strcpy(tmp, prefix), ".sa");  // partial suffix array (SA)
	bwt_restore_sa(tmp, bwt);
	free(tmp); free(prefix);
	return bwt;
}

bwaidx_t *bwa_idx_load_from_disk(const char *hint, int which)
{
	bwaidx_t *idx;
	char *prefix;
    //fprintf(stderr, "%s hint  = %s\n", __func__, hint);
	prefix = bwa_idx_infer_prefix(hint);
    //fprintf(stderr, "%s prefix = %s\n", __func__, prefix);
	if (prefix == 0) {
        fprintf(stderr, "hello.\n");
        fprintf(stderr, "hint: %s\n", hint);

		fprintf(stderr, "[E::%s] fail to locate the index files\n", __func__);
		return 0;
	}
	idx = (bwaidx_t*)calloc(1, sizeof(bwaidx_t));
	if (which & BWA_IDX_BWT) idx->bwt = bwa_idx_load_bwt(hint);
	if (which & BWA_IDX_BNS) {
		int i, c;
		idx->bns = bns_restore(prefix);
		for (i = c = 0; i < idx->bns->n_seqs; ++i)
			if (idx->bns->anns[i].is_alt) ++c;
            std::cerr << "* read " << c << " ALT contigs\n";
		if (which & BWA_IDX_PAC) {
			idx->pac = (uint8_t*)calloc(idx->bns->l_pac/4+1, 1);
            // FIXME error handling
			fread(idx->pac, 1, idx->bns->l_pac/4+1, idx->bns->fp_pac); // concatenated 2-bit encoded sequence
			fclose(idx->bns->fp_pac);
			idx->bns->fp_pac = 0;
		}
	}
	free(prefix);
	return idx;
}

bwaidx_t *bwa_idx_load(const char *hint, int which)
{
	return bwa_idx_load_from_disk(hint, which);
}

void bwa_idx_destroy(bwaidx_t *idx)
{
	if (idx == 0) return;
	if (idx->mem == 0) {
		if (idx->bwt) bwt_destroy(idx->bwt);
		if (idx->bns) bns_destroy(idx->bns);
		if (idx->pac) free(idx->pac);
	} else {
		free(idx->bwt); free(idx->bns->anns); free(idx->bns);
		if (!idx->is_shm) free(idx->mem);
	}
	free(idx);
}


static char *bwa_escape(char *s)
{
	char *p, *q;
	for (p = q = s; *p; ++p) {
		if (*p == '\\') {
			++p;
			if (*p == 't') *q++ = '\t';
			else if (*p == 'n') *q++ = '\n';
			else if (*p == 'r') *q++ = '\r';
			else if (*p == '\\') *q++ = '\\';
		} else *q++ = *p;
	}
	*q = '\0';
	return s;
}

char *bwa_insert_header(const char *s, char *hdr)
{
	int len = 0;
	if (s == 0 || s[0] != '@') return hdr;
	if (hdr) {
		len = strlen(hdr);
		hdr = (char*)realloc(hdr, len + strlen(s) + 2);
		hdr[len++] = '\n';
		strcpy(hdr + len, s);
	} else hdr = strdup(s);
	bwa_escape(hdr + len);
	return hdr;
}



int main_mem(int argc, char *argv[])
{
	mem_opt_t *opt, opt0;
	int fd, fd2, i, c, ignore_alt = 0, no_mt_io = 0;
	int fixed_chunk_size = -1;
	gzFile fp, fp2 = 0;
	char *p, *rg_line = 0, *hdr_line = 0;
	const char *mode = 0;
	void *ko = 0, *ko2 = 0;
	mem_pestat_t pes[4];
	ktp_aux_t aux;

	memset(&aux, 0, sizeof(ktp_aux_t));
    std::ofstream samfile;
    std::string samfilepath;
    aux.samout = &std::cout;

	memset(pes, 0, 4 * sizeof(mem_pestat_t));
	for (i = 0; i < 4; ++i) pes[i].failed = 1;

    int loading_factor = 1;
    g3_opt_t *g3_opt = g3_opt_init();
	aux.opt = opt = mem_opt_init();
	memset(&opt0, 0, sizeof(mem_opt_t));
	while ((c = getopt(argc, argv, "51qpbaMCSPVYjuk:c:v:s:r:t:R:A:B:O:E:U:w:L:d:F:T:Q:D:m:I:N:o:f:W:x:G:g:l:h:y:K:X:H:Z:")) >= 0) {
		if (c == 'k') opt->min_seed_len = atoi(optarg), opt0.min_seed_len = 1;
		else if (c == 'b') g3_opt->baseline = 1;
		else if (c == '1') no_mt_io = 1;
		else if (c == 'x') mode = optarg;
		else if (c == 'F') loading_factor = atoi(optarg); 
		else if (c == 'w') opt->w = atoi(optarg), opt0.w = 1;
		else if (c == 'A') opt->a = atoi(optarg), opt0.a = 1;
		else if (c == 'B') opt->b = atoi(optarg), opt0.b = 1;
		else if (c == 'T') opt->T = atoi(optarg), opt0.T = 1;
		else if (c == 'U') opt->pen_unpaired = atoi(optarg), opt0.pen_unpaired = 1;
		else if (c == 't') opt->n_threads = atoi(optarg), opt->n_threads = opt->n_threads > 1? opt->n_threads : 1;
		else if (c == 'P') opt->flag |= MEM_F_NOPAIRING;
		else if (c == 'a') opt->flag |= MEM_F_ALL;
		else if (c == 'p') opt->flag |= MEM_F_PE | MEM_F_SMARTPE;
		else if (c == 'M') opt->flag |= MEM_F_NO_MULTI;
		else if (c == 'S') opt->flag |= MEM_F_NO_RESCUE;
		else if (c == 'Y') opt->flag |= MEM_F_SOFTCLIP;
		else if (c == 'V') opt->flag |= MEM_F_REF_HDR;
		else if (c == '5') opt->flag |= MEM_F_PRIMARY5 | MEM_F_KEEP_SUPP_MAPQ; // always apply MEM_F_KEEP_SUPP_MAPQ with -5
		else if (c == 'q') opt->flag |= MEM_F_KEEP_SUPP_MAPQ;
		else if (c == 'u') opt->flag |= MEM_F_XB;
		else if (c == 'c') opt->max_occ = atoi(optarg), opt0.max_occ = 1;
		else if (c == 'd') opt->zdrop = atoi(optarg), opt0.zdrop = 1;
		else if (c == 'v') g3_opt->verbosity = atoi(optarg);
		else if (c == 'j') ignore_alt = 1;
		else if (c == 'r') opt->split_factor = atof(optarg), opt0.split_factor = 1.;
		else if (c == 'D') opt->drop_ratio = atof(optarg), opt0.drop_ratio = 1.;
		else if (c == 'm') opt->max_matesw = atoi(optarg), opt0.max_matesw = 1;
		else if (c == 's') opt->split_width = atoi(optarg), opt0.split_width = 1;
		else if (c == 'G') opt->max_chain_gap = atoi(optarg), opt0.max_chain_gap = 1;
		else if (c == 'g'){
            g3_opt->num_use_gpus = atoi(optarg);
        }
		else if (c == 'l'){
            g3_opt->print_mask = atoi(optarg);
        } 
		else if (c == 'Z'){
            g3_opt->batch_size = atoi(optarg);
        } 
		else if (c == 'N') opt->max_chain_extend = atoi(optarg), opt0.max_chain_extend = 1;
        else if (c == 'o' || c == 'f') {
            samfilepath = optarg;
            samfile.open(samfilepath);
            if(!samfile.is_open()){
                std::cerr << "ERROR. Could not open output filepath "
                    << samfilepath << std::endl;
                exit(EXIT_FAILURE);
            }
            aux.samout = &samfile;
        }
		else if (c == 'W') opt->min_chain_weight = atoi(optarg), opt0.min_chain_weight = 1;
		else if (c == 'y') opt->max_mem_intv = atol(optarg), opt0.max_mem_intv = 1;
		else if (c == 'C') aux.copy_comment = 1;
		else if (c == 'K') fixed_chunk_size = atoi(optarg);
		else if (c == 'X') opt->mask_level = atof(optarg);
		else if (c == 'h') {
			opt0.max_XA_hits = opt0.max_XA_hits_alt = 1;
			opt->max_XA_hits = opt->max_XA_hits_alt = strtol(optarg, &p, 10);
			if (*p != 0 && ispunct(*p) && isdigit(p[1]))
				opt->max_XA_hits_alt = strtol(p+1, &p, 10);
		}
		else if (c == 'Q') {
			opt0.mapQ_coef_len = 1;
			opt->mapQ_coef_len = atoi(optarg);
			opt->mapQ_coef_fac = opt->mapQ_coef_len > 0? log(opt->mapQ_coef_len) : 0;
		} else if (c == 'O') {
			opt0.o_del = opt0.o_ins = 1;
			opt->o_del = opt->o_ins = strtol(optarg, &p, 10);
			if (*p != 0 && ispunct(*p) && isdigit(p[1]))
				opt->o_ins = strtol(p+1, &p, 10);
		} else if (c == 'E') {
			opt0.e_del = opt0.e_ins = 1;
			opt->e_del = opt->e_ins = strtol(optarg, &p, 10);
			if (*p != 0 && ispunct(*p) && isdigit(p[1]))
				opt->e_ins = strtol(p+1, &p, 10);
		} else if (c == 'L') {
			opt0.pen_clip5 = opt0.pen_clip3 = 1;
			opt->pen_clip5 = opt->pen_clip3 = strtol(optarg, &p, 10);
			if (*p != 0 && ispunct(*p) && isdigit(p[1]))
				opt->pen_clip3 = strtol(p+1, &p, 10);
		} else if (c == 'R') {
			//if ((rg_line = bwa_set_rg(optarg)) == 0) return 1; // FIXME: memory leak
		} else if (c == 'H') {
			if (optarg[0] != '@') {
				FILE *fp;
				if ((fp = fopen(optarg, "r")) != 0) {
					char *buf;
					buf = (char*)calloc(1, 0x10000);
					while (fgets(buf, 0xffff, fp)) {
						i = strlen(buf);
						assert(buf[i-1] == '\n'); // a long line
						buf[i-1] = 0;
						hdr_line = bwa_insert_header(buf, hdr_line);
					}
					free(buf);
					fclose(fp);
				}
			} else hdr_line = bwa_insert_header(optarg, hdr_line);
		} else if (c == 'I') { // specify the insert size distribution
			aux.pes0 = pes;
			pes[1].failed = 0;
			pes[1].avg = strtod(optarg, &p);
			pes[1].std = pes[1].avg * .1;
			if (*p != 0 && ispunct(*p) && isdigit(p[1]))
				pes[1].std = strtod(p+1, &p);
			pes[1].high = (int)(pes[1].avg + 4. * pes[1].std + .499);
			pes[1].low  = (int)(pes[1].avg - 4. * pes[1].std + .499);
			if (pes[1].low < 1) pes[1].low = 1;
			if (*p != 0 && ispunct(*p) && isdigit(p[1]))
				pes[1].high = (int)(strtod(p+1, &p) + .499);
			if (*p != 0 && ispunct(*p) && isdigit(p[1]))
				pes[1].low  = (int)(strtod(p+1, &p) + .499);
				fprintf(stderr, "[M::%s] mean insert size: %.3f, stddev: %.3f, max: %d, min: %d\n",
						__func__, pes[1].avg, pes[1].std, pes[1].high, pes[1].low);
		}
		else return 1;
	}

	if (rg_line) {
		hdr_line = bwa_insert_header(rg_line, hdr_line);
		free(rg_line);
	}

	if (opt->n_threads < 1) opt->n_threads = 1;
	if (optind + 1 >= argc || optind + 3 < argc) {
		fprintf(stderr, "\n");
		fprintf(stderr, "Usage: bwa mem [options] <idxbase> <hashtable> <in1.fq> [in2.fq]\n\n");
		fprintf(stderr, "Algorithm options:\n\n");
		fprintf(stderr, "       -t INT        number of threads [%d]\n", opt->n_threads);
		fprintf(stderr, "       -k INT        minimum seed length [%d]\n", opt->min_seed_len);
		fprintf(stderr, "       -w INT        band width for banded alignment [%d]\n", opt->w);
		fprintf(stderr, "       -d INT        off-diagonal X-dropoff [%d]\n", opt->zdrop);
		fprintf(stderr, "       -r FLOAT      look for internal seeds inside a seed longer than {-k} * FLOAT [%g]\n", opt->split_factor);
		fprintf(stderr, "       -y INT        seed occurrence for the 3rd round seeding [%ld]\n", (long)opt->max_mem_intv);
//		fprintf(stderr, "       -s INT        look for internal seeds inside a seed with less than INT occ [%d]\n", opt->split_width);
		fprintf(stderr, "       -c INT        skip seeds with more than INT occurrences [%d]\n", opt->max_occ);
		fprintf(stderr, "       -D FLOAT      drop chains shorter than FLOAT fraction of the longest overlapping chain [%.2f]\n", opt->drop_ratio);
		fprintf(stderr, "       -W INT        discard a chain if seeded bases shorter than INT [0]\n");
		fprintf(stderr, "       -m INT        perform at most INT rounds of mate rescues for each read [%d]\n", opt->max_matesw);
		fprintf(stderr, "       -S            skip mate rescue\n");
		fprintf(stderr, "       -P            skip pairing; mate rescue performed unless -S also in use\n");
		fprintf(stderr, "\nScoring options:\n\n");
		fprintf(stderr, "       -A INT        score for a sequence match, which scales options -TdBOELU unless overridden [%d]\n", opt->a);
		fprintf(stderr, "       -B INT        penalty for a mismatch [%d]\n", opt->b);
		fprintf(stderr, "       -O INT[,INT]  gap open penalties for deletions and insertions [%d,%d]\n", opt->o_del, opt->o_ins);
		fprintf(stderr, "       -E INT[,INT]  gap extension penalty; a gap of size k cost '{-O} + {-E}*k' [%d,%d]\n", opt->e_del, opt->e_ins);
		fprintf(stderr, "       -L INT[,INT]  penalty for 5'- and 3'-end clipping [%d,%d]\n", opt->pen_clip5, opt->pen_clip3);
		fprintf(stderr, "       -U INT        penalty for an unpaired read pair [%d]\n\n", opt->pen_unpaired);
		fprintf(stderr, "       -x STR        read type. Setting -x changes multiple parameters unless overridden [null]\n");
		fprintf(stderr, "                     pacbio: -k17 -W40 -r10 -A1 -B1 -O1 -E1 -L0  (PacBio reads to ref)\n");
		fprintf(stderr, "                     ont2d: -k14 -W20 -r10 -A1 -B1 -O1 -E1 -L0  (Oxford Nanopore 2D-reads to ref)\n");
		fprintf(stderr, "                     intractg: -B9 -O16 -L5  (intra-species contigs to ref)\n");
		fprintf(stderr, "\nInput/output options:\n\n");
		fprintf(stderr, "       -p            smart pairing (ignoring in2.fq)\n");
		fprintf(stderr, "       -R STR        read group header line such as '@RG\\tID:foo\\tSM:bar' [null]\n");
		fprintf(stderr, "       -H STR/FILE   insert STR to header if it starts with @; or insert lines in FILE [null]\n");
		fprintf(stderr, "       -o FILE       sam file to output results to [stdout]\n");
		fprintf(stderr, "       -j            treat ALT contigs as part of the primary assembly (i.e. ignore <idxbase>.alt file)\n");
		fprintf(stderr, "       -5            for split alignment, take the alignment with the smallest coordinate as primary\n");
		fprintf(stderr, "       -q            don't modify mapQ of supplementary alignments\n");
		fprintf(stderr, "       -K INT        process INT input bases in each batch regardless of nThreads (for reproducibility) []\n");
		fprintf(stderr, "\n");
		fprintf(stderr, "       -v INT        verbosity level: 1=error, 2=warning, 3=message, 4+=debugging []\n");
		fprintf(stderr, "       -T INT        minimum score to output [%d]\n", opt->T);
		fprintf(stderr, "       -h INT[,INT]  if there are <INT hits with score >80%% of the max score, output all in XA [%d,%d]\n", opt->max_XA_hits, opt->max_XA_hits_alt);
		fprintf(stderr, "       -a            output all alignments for SE or unpaired PE\n");
		fprintf(stderr, "       -C            append FASTA/FASTQ comment to SAM output\n");
		fprintf(stderr, "       -V            output the reference FASTA header in the XR tag\n");
		fprintf(stderr, "       -Y            use soft clipping for supplementary alignments\n");
		fprintf(stderr, "       -M            mark shorter split hits as secondary\n\n");
		fprintf(stderr, "       -I FLOAT[,FLOAT[,INT[,INT]]]\n");
		fprintf(stderr, "                     specify the mean, standard deviation (10%% of the mean if absent), max\n");
		fprintf(stderr, "                     (4 sigma from the mean if absent) and min of the insert size distribution.\n");
		fprintf(stderr, "                     FR orientation only. [inferred]\n");
		fprintf(stderr, "\n");
		fprintf(stderr, "Note: Please read the man page for detailed description of the command line and options.\n");
		fprintf(stderr, "\n");
		free(opt);
		return 1;
	}

	if (mode) {
		if (strcmp(mode, "intractg") == 0) {
			if (!opt0.o_del) opt->o_del = 16;
			if (!opt0.o_ins) opt->o_ins = 16;
			if (!opt0.b) opt->b = 9;
			if (!opt0.pen_clip5) opt->pen_clip5 = 5;
			if (!opt0.pen_clip3) opt->pen_clip3 = 5;
		} else if (strcmp(mode, "pacbio") == 0 || strcmp(mode, "pbref") == 0 || strcmp(mode, "ont2d") == 0) {
			if (!opt0.o_del) opt->o_del = 1;
			if (!opt0.e_del) opt->e_del = 1;
			if (!opt0.o_ins) opt->o_ins = 1;
			if (!opt0.e_ins) opt->e_ins = 1;
			if (!opt0.b) opt->b = 1;
			if (opt0.split_factor == 0.) opt->split_factor = 10.;
			if (strcmp(mode, "ont2d") == 0) {
				if (!opt0.min_chain_weight) opt->min_chain_weight = 20;
				if (!opt0.min_seed_len) opt->min_seed_len = 14;
				if (!opt0.pen_clip5) opt->pen_clip5 = 0;
				if (!opt0.pen_clip3) opt->pen_clip3 = 0;
			} else {
				if (!opt0.min_chain_weight) opt->min_chain_weight = 40;
				if (!opt0.min_seed_len) opt->min_seed_len = 17;
				if (!opt0.pen_clip5) opt->pen_clip5 = 0;
				if (!opt0.pen_clip3) opt->pen_clip3 = 0;
			}
		} else {
			fprintf(stderr, "[E::%s] unknown read type '%s'\n", __func__, mode);
			return 1; // FIXME memory leak
		}
	} else update_a(opt, &opt0);
	bwa_fill_scmat(opt->a, opt->b, opt->mat);

    if ((aux.idx = bwa_idx_load(argv[optind], BWA_IDX_ALL)) == 0) return 1; // FIXME: memory leak

    // load occ2
	FMI_wrapper *obj = FMI_wrapper_create(argv[optind]);
    FMI_wrapper_load_index(obj, &(aux.loadedIndex));

	// load kmer hash
	aux.kmerHashTab = loadKMerIndex(argv[optind++]);
    fprintf(stderr, "* kmer hash loaded\n");
    fprintf(stderr, "* all idx loaded\n");

	if (ignore_alt)
		for (i = 0; i < aux.idx->bns->n_seqs; ++i)
			aux.idx->bns->anns[i].is_alt = 0;

    // initiate read loader
	ko = kopen(argv[optind], &fd);
	if (ko == 0) {
		 fprintf(stderr, "[E::%s] fail to open file `%s'.\n", __func__, argv[optind]);
		return 1;
	}
	fp = gzdopen(fd, "r");
	aux.ks = kseq_init(fp);
    ++optind;

    // initiate read2 loader (if given)
	if (optind < argc) {
		if (opt->flag&MEM_F_PE) {
				fprintf(stderr, "[W::%s] when '-p' is in use, the second query file is ignored.\n", __func__);
		} else {
			ko2 = kopen(argv[optind], &fd2);
			if (ko2 == 0) {
			 fprintf(stderr, "[E::%s] fail to open file `%s'.\n", __func__, argv[optind]);
				return 1;
			}
			fp2 = gzdopen(fd2, "r");
			aux.ks2 = kseq_init(fp2);
			opt->flag |= MEM_F_PE;
		}
	}

	//bwa_print_sam_hdr(aux.idx->bns, hdr_line); 

    // storage loading batch size
	//aux.actual_loading_batch_size = fixed_loading_batch_size > 0? fixed_loading_batch_size : opt->loading_batch_size * opt->n_threads;
	aux.loading_batch_size = g3_opt->batch_size * g3_opt->num_use_gpus * loading_factor;
    fprintf(stderr, "* per GPU batch size = %d\n", g3_opt->batch_size);
    fprintf(stderr, "* storage loading factor %d\n", loading_factor);
    fprintf(stderr, "* storage loading batch size %ld\n", aux.loading_batch_size);


    struct timespec start, end;
    double walltime;

    clock_gettime(CLOCK_REALTIME,  &start);
	pipeline(&aux, g3_opt);
    clock_gettime(CLOCK_REALTIME,  &end);

    walltime = (end.tv_sec - start.tv_sec) +\
                           (end.tv_nsec - start.tv_nsec) / 1e9;
    fprintf(stderr,"* Wall-clock time after index loading: %.2lf seconds\n\n\n", walltime);

    std::cerr << "	setting up GPUs (alloc & idx memcpy): ";
    std::cerr << std::fixed << std::setprecision(2)
        << tprof[0][GPU_SETUP] / 1000 << " seconds" << std::endl;

    std::cerr << "	loading the first loading batch: ";
    std::cerr << std::fixed << std::setprecision(2)
        << tprof[0][FILE_INPUT_FIRST] / 1000 << " seconds" << std::endl;

    // Runtime profiling stats
    //  0: min, 1: avg, 2: max.
    float walltime_seeding[3], walltime_chaining[3], walltime_extending[3];
    //  0: seeding, 1: chaining, 2: extending. 3: total, 4: push, 5: pull.
    float tim, min_tim[8], max_tim[8], sum_tim[8];
    for(int k=0; k<6; k++){
        min_tim[k] = FLT_MAX; max_tim[k] = FLT_MIN; sum_tim[k] = 0;
    }
    float *tims;
    for(int gpuid = 0; gpuid < g3_opt->num_use_gpus; gpuid++){
        tims = tprof[gpuid];

        tim = tims[S_SMEM] + tims[S_R2] + tims[S_R3];
        if(tim < min_tim[0]) min_tim[0] = tim;
        if(tim > max_tim[0]) max_tim[0] = tim;
        sum_tim[0] += tim;

        tim = tims[C_SAL] + tims[C_SORT_SEEDS] + tims[C_CHAIN]
            + tims[C_SORT_CHAINS] + tims[C_FILTER];
        if(tim < min_tim[1]) min_tim[1] = tim;
        if(tim > max_tim[1]) max_tim[1] = tim;
        sum_tim[1] += tim;

        tim = tims[E_PAIRGEN] + tims[E_EXTEND] + tims[E_FILTER_MARK]
            + tims[E_SORT_ALNS] + tims[E_T_PAIRGEN] + tims[E_TRACEBACK]
            + tims[E_FINALIZE];
        if(tim < min_tim[2]) min_tim[2] = tim;
        if(tim > max_tim[2]) max_tim[2] = tim;
        sum_tim[2] += tim;

        tim = tims[COMPUTE_TOTAL];
        if(tim < min_tim[3]) min_tim[3] = tim;
        if(tim > max_tim[3]) max_tim[3] = tim;
        sum_tim[3] += tim;

        tim = tims[PUSH_TOTAL];
        if(tim < min_tim[4]) min_tim[4] = tim;
        if(tim > max_tim[4]) max_tim[4] = tim;
        sum_tim[4] += tim;

        tim = tims[PULL_TOTAL];
        if(tim < min_tim[5]) min_tim[5] = tim;
        if(tim > max_tim[5]) max_tim[5] = tim;
        sum_tim[5] += tim;

    }
    std::cerr << "	\t- compute total: (";
    std::cerr << std::fixed << std::setprecision(2)
        << sum_tim[3] / g3_opt->num_use_gpus / 1000 << ", "
        << min_tim[3] / 1000 << ", "
        << max_tim[3] / 1000 << ") seconds" << std::endl;

    std::cerr << "	\t\t- seeding: (";
    std::cerr << std::fixed << std::setprecision(2)
        << sum_tim[0] / g3_opt->num_use_gpus / 1000 << ", "
        << min_tim[0] / 1000 << ", "
        << max_tim[0] / 1000 << ") seconds" << std::endl;

    std::cerr << "	\t\t- chaining: (";
    std::cerr << std::fixed << std::setprecision(2)
        << sum_tim[1] / g3_opt->num_use_gpus / 1000 << ", "
        << min_tim[1] / 1000 << ", "
        << max_tim[1] / 1000 << ") seconds" << std::endl;

    std::cerr << "	\t\t- extending: (";
    std::cerr << std::fixed << std::setprecision(2)
        << sum_tim[2] / g3_opt->num_use_gpus / 1000 << ", "
        << min_tim[2] / 1000 << ", "
        << max_tim[2] / 1000 << ") seconds" << std::endl;

    std::cerr << "	\t- push: (";
    std::cerr << std::fixed << std::setprecision(2)
        << sum_tim[4] / g3_opt->num_use_gpus / 1000 << ", "
        << min_tim[4] / 1000 << ", "
        << max_tim[4] / 1000 << ") seconds" << std::endl;

    std::cerr << "	\t- pull: (";
    std::cerr << std::fixed << std::setprecision(2)
        << sum_tim[5] / g3_opt->num_use_gpus / 1000 << ", "
        << min_tim[5] / 1000 << ", "
        << max_tim[5] / 1000 << ") seconds" << std::endl;

    std::cerr << "	\t---------------------------" << std::endl;
    std::cerr << "	 In total , FILE input ";
    std::cerr << std::fixed << std::setprecision(3)
        << tprof[0][FILE_INPUT] / 1000 << " vs ALIGN "
        << tprof[0][ALIGNER_TOP] / 1000 << " vs FILE output "
        << tprof[0][FILE_OUTPUT] / 1000 << " seconds"
        << std::endl << std::endl;


	FMI_wrapper_destroy(obj);
	free(hdr_line);
	free(opt);
    free(g3_opt);
	bwa_idx_destroy(aux.idx);
	kseq_destroy(aux.ks);
	gzclose(fp); kclose(ko);
	if (aux.ks2) {
		kseq_destroy(aux.ks2);
		gzclose(fp2); kclose(ko2);
	}
	return 0;
}
