#include <assert.h>
#include "UpixBaseLayer.h"

void UpixReduceMaxLayer::forward_float()
{
	assert(_input.shape.w == _output.shape.w);
	assert(_input.shape.h == _output.shape.h);
	assert(_output.shape.c == 1);
	const float* inptr = _input.fdata.data();
	int i, j;
	for (j = 0; j < _output.shape.wh; j++, inptr++) {
		float maxv = -FLT_MAX;
		for (i = 0; i < _input.shape.c; i++) {
			maxv = std::max(maxv, inptr[i*_output.shape.wh]);
		}
		_output.fdata[j] = maxv;
	}
}

void UpixReduceMaxLayer::forward_int()
{
	assert(_input.shape.w == _output.shape.w);
	assert(_input.shape.h == _output.shape.h);
	assert(_output.shape.c == 1);
	const int* inptr = _input.idata.data();
	int i, j;
	for (j = 0; j < _output.shape.wh; j++, inptr++) {
		int maxv = -INT_MAX;
		for (i = 0; i < _input.shape.c; i++) {
			maxv = std::max(maxv, inptr[i * _output.shape.wh]);
		}
		_output.idata[j] = maxv;
	}
}