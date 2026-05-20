#include <assert.h>
#include "UpixBaseLayer.h"

template<typename T>
static void global_avg(const std::vector<T>& input, std::vector<T>& output, int wh, int c)
{
	for (int i = 0; i < c; i++) {
		T sum = 0;
		for (int j = 0; j < wh; j++) {
			sum += input[i * wh + j];
		}
		output[i] = (sum / wh);
	}
}

void UpixGAvgLayer::forward_float()
{
	assert(_output.shape.w == 1);
	assert(_output.shape.h == 1);
	assert(_output.shape.c == _input.shape.c);

	global_avg<float>(_input.fdata, _output.fdata, _input.shape.wh, _input.shape.c);
}

void UpixGAvgLayer::forward_int()
{
	assert(_output.shape.w == 1);
	assert(_output.shape.h == 1);
	assert(_output.shape.c == _input.shape.c);

	global_avg<int>(_input.idata, _output.idata, _input.shape.wh, _input.shape.c);
}
