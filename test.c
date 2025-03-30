#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NUM_SAMPLES 8
#define GIGA (1000000000)

int main(int argc, char *argv[]){
    int request_size;
    int *p;
    int num_samples;
    int offset, stride;
    if(argc < 2){
        fprintf(stderr, "usage: <prog> <request_size in GB>\n");
        exit(EXIT_FAILURE);
    }

    request_size = atoi(argv[1]);
    fprintf(stderr, "requested size: %d GB\n", request_size);
    fprintf(stderr, "              : %ld in Bytes\n", (long int)request_size * GIGA);
    p = (int*)malloc(request_size * GIGA);
    if(p==0){
        fprintf(stderr, "malloc failed\n");
        exit(EXIT_FAILURE);
    }

    num_samples = NUM_SAMPLES;
    stride = request_size / num_samples;
    offset = 0;
    if(stride==0){
        fprintf(stdout, "writing %4d to offset %8d: %d\n", rand(), offset, p[offset]);
        fprintf(stdout, "offset %8d: %d\n", offset, p[offset]);
    } else{
        for(; offset < request_size; offset += stride){
            fprintf(stdout, "writing %4d to offset %8d: %d\n", rand(), offset, p[offset]);
        }
        for(offset = 0; offset < request_size; offset += stride){
            fprintf(stdout, "offset %8d: %d\n", offset, p[offset]);
        }
    }

    free(p);
    exit(EXIT_SUCCESS);
}
