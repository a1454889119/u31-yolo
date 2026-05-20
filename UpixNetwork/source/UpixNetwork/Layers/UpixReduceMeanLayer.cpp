#include <assert.h>
#include "UpixBaseLayer.h"

void UpixReduceMeanLayer::forward_float()
{
	assert(_input.shape.w == _output.shape.w);
	assert(_input.shape.h == _output.shape.h);
	assert(_output.shape.c == 1);
	const float* inptr = _input.fdata.data();
	int i, j;
	for (j = 0; j < _output.shape.wh; j++, inptr++) {
		float sum = 0.0f;
		for (i = 0; i < _input.shape.c; i++) {
			sum += inptr[i * _output.shape.wh];
		}
		_output.fdata[j] = sum / _input.shape.c;
	}
}

void UpixReduceMeanLayer::forward_int()
{
	assert(_input.shape.w == _output.shape.w);
	assert(_input.shape.h == _output.shape.h);
	assert(_output.shape.c == 1);
	const int* inptr = _input.idata.data();
	int i, j;
	for (j = 0; j < _output.shape.wh; j++, inptr++) {
		int sum = 0;
		for (i = 0; i < _input.shape.c; i++) {
			sum += inptr[i * _output.shape.wh];
		}
		_output.idata[j] = sum / _input.shape.c;
	}
}