#include <assert.h>
#include <memory.h>
#include "UpixBaseLayer.h"

template<typename T>
static void shuffle(const std::vector<T>& input, std::vector<T>& output, const TensorShape& shape)
{
	const T* in_ptr = input.data();
	T* out_ptr0 = output.data();
	T* out_ptr1 = out_ptr0 + shape.wh * (shape.c / 2);


	int i;
	for (i = 0; i < shape.c; i += 2) {
		memcpy(out_ptr0, in_ptr, shape.wh * sizeof(T));
		in_ptr += shape.wh;
		out_ptr0 += shape.wh;

		memcpy(out_ptr1, in_ptr, shape.wh * sizeof(T));
		in_ptr += shape.wh;
		out_ptr1 += shape.wh;
	}
}

void UpixShuffleLayer::forward_float()
{
	assert(_input.shape.w == _output.shape.w);
	assert(_input.shape.h == _output.shape.h);
	assert(_input.shape.c == _output.shape.c);
	assert(_input.shape.c % 2 == 0);

	shuffle<float>(_input.fdata, _output.fdata, _input.shape);
}

void UpixShuffleLayer::forward_int()
{
	assert(_input.shape.w == _output.shape.w);
	assert(_input.shape.h == _output.shape.h);
	assert(_input.shape.c == _output.shape.c);
	assert(_input.shape.c % 2 == 0);

	shuffle<int>(_input.idata, _output.idata, _input.shape);
}
