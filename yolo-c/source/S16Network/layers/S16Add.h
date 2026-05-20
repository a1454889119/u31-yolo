#ifndef __S16_ADD_LAYER_H__
#define __S16_ADD_LAYER_H__

#include "S16BaseLayer.h"

typedef struct stS16AddLayer {
	ActivateType activate_type;
}S16AddLayer;

CAPI void S16_AddForward(void* userdata, const S16Tensor* input, S16Tensor* output);

#endif // !__S16_ADD_LAYER_H__
