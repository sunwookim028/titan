#ifndef _KOPEN_H
#define _KOPEN_H

#ifdef __cplusplus
extern "C" {
#endif

    void *kopen(const char *fn, int *_fd);
    int kclose(void *a);

#ifdef __cplusplus
}
#endif

#endif
