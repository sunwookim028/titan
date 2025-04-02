#ifndef KSEQ_WRAPPER_H
#define KSEQ_WRAPPER_H

#include <zlib.h>          // Ensure zlib types and functions (e.g. gzFile, gzread) are defined
KSEQ_INIT(gzFile, gzread)
#include "kseq.h"          // This will define kseq_t and inline versions of kseq_read, kseq_init, etc.
                           //
#endif // KSEQ_WRAPPER_H
