#include <assert.h>
#include <memory.h>
#include "UpixBaseLayer.h"

template<typename T>
static void upsample(const std::vector<T>& input, std::vector<T>& output, const TensorShape& shape, int ksize)
{
	const T* in_ptr = input.data();
	T* out_ptr = output.data();
	int outw = shape.w * ksize;
	int i, j, k;
	for (i = 0; i < shape.c; i++) {
		for (j = 0; j < shape.h; j++) {
			int t;
			T* out_first = out_ptr;

			for (k = 0; k < shape.w; k++) {
				for (t = 0; t < ksize; t++) {
					*out_ptr++ = *in_ptr;
				}
				in_ptr++;
			}

			for (t = 1; t < ksize; t++) {
				memcpy(out_ptr, out_first, sizeof(T) * outw);
				out_ptr += outw;
			}
		}
	}
}

void UpixUpsampleLayer::forward_float()
{
	assert(_input.shape.w * _info->ksize == _output.shape.w);
	assert(_input.shape.h * _info->ksize == _output.shape.h);
	assert(_input.shape.c == _output.shape.c);

	upsample<float>(_input.fdata, _output.fdata, _input.shape, _info->ksize);
}

void UpixUpsampleLayer::forward_int()
{
	assert(_input.shape.w * _info->ksize == _output.shape.w);
	assert(_input.shape.h * _info->ksize == _output.shape.h);
	assert(_input.shape.c == _output.shape.c);

	upsample<int>(_input.idata, _output.idata, _input.shape, _info->ksize);
}