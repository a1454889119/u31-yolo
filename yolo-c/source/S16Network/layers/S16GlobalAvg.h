#ifndef __S16_GLOBAL_AVG_LAYER_H__
#define __S16_GLOBAL_AVG_LAYER_H__

#include "S16BaseLayer.h"

typedef struct stS16GlobalAvgLayer {
	int dummy;
}S16GlobalAvgLayer;

CAPI void S16_GlobalAvgForward(void* userdata, const S16Tensor* input, S16Tensor* output);

#endif // !__S16_GLOBAL_AVG_LAYER_H__
