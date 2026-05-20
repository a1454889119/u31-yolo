#include <assert.h>
#include "UpixBaseLayer.h"

void UpixAddLayer::forward_float()
{
	int count = _output.shape.whc;
	int parts = _input.shape.c / _output.shape.c;
	assert(_input.shape.w == _output.shape.w);
	assert(_input.shape.h == _output.shape.h);
	assert(parts > 1 && parts * _output.shape.c == _input.shape.c);

	for (int i = 0; i < count; i++) {
		_output.fdata[i] = 0.0f;
		for (int j = 0; j < parts; j++) {
			_output.fdata[i] += _input.fdata[i + j * count];
		}
	}
}


void UpixAddLayer::forward_int()
{
	int count = _input.shape.whc;
	int parts = _input.shape.c / _output.shape.c;
	assert(_input.shape.w == _output.shape.w);
	assert(_input.shape.h == _output.shape.h);
	assert(parts > 1 && parts * _output.shape.c == _input.shape.c);

	for (int i = 0; i < count; i++) {
		_output.idata[i] = 0;
		for (int j = 0; j < parts; j++) {
			_output.idata[i] += _input.idata[i + j * count];
		}
	}
}