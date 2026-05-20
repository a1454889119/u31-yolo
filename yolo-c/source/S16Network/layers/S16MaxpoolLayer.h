#ifndef __S16_MAXPOOL_LAYER_H__
#define __S16_MAXPOOL_LAYER_H__

#include "S16BaseLayer.h"

typedef struct stS16MaxpoolLayer {
	int size;
	int stride;
	int pad;
}S16MaxpoolLayer;

CAPI void S16_MaxpoolForward (void *userdata, const S16Tensor *input, S16Tensor *output);

#endif