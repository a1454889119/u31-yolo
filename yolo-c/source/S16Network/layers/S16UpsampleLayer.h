#ifndef __S16_UPSAMPLE_LAYER_H__
#define __S16_UPSAMPLE_LAYER_H__

#include "S16BaseLayer.h"

typedef struct stS16UpsampleLayer {
	int ksize;
}S16UpsampleLayer;

CAPI void S16_UpsampleForward(void* userdata, const S16Tensor* input, S16Tensor* output);

#endif