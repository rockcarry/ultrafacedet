#ifndef __ULTRAFACEDET_H__
#define __ULTRAFACEDET_H__

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float score;
    int   x1, y1, x2, y2;
} BBOX;

void* facedet_init  (char *path, int inw, int inh);
void  facedet_free  (void *ctxt);
int   facedet_detect(void *ctxt, BBOX *bboxlist, int n, uint8_t *bitmap);

#ifdef __cplusplus
}
#endif

#endif

