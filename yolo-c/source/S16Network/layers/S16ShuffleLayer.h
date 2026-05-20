#ifndef __S16_SHUFFLE_LAYER_H__
#define __S16_SHUFFLE_LAYER_H__

#include "S16BaseLayer.h"

typedef struct stS16ShuffleLayer {
	int dummy;
}S16ShuffleLayer;

CAPI void S16_ShuffleForward(void* userdata, const S16Tensor* input, S16Tensor* output);

#endif