#ifndef __S16_CONVOLUTION_LAYER_H__
#define __S16_CONVOLUTION_LAYER_H__

#include "S16BaseLayer.h"

typedef struct stS16ConvLayer {
	ActivateType activate_type;
	int groups;
	int ksize;
	int stride;
	int pad;

	int shift;

	s16 *kernels;
	int *biases;
}S16ConvLayer;

CAPI void S16_ConvolutionForward (void *userdata, const S16Tensor *input, S16Tensor *output);

#endif