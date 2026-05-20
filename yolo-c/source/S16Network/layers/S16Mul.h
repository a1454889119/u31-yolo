#ifndef __S16_MUL_LAYER_H__
#define __S16_MUL_LAYER_H__

#include "S16BaseLayer.h"

typedef struct stS16MulLayer {
	int dummy;
}S16MulLayer;

CAPI void S16_MulForward(void* userdata, const S16Tensor* input, S16Tensor* output);

#endif // !__S16_MUL_LAYER_H__
