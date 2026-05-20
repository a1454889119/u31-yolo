#include <assert.h>
#include "UpixBaseLayer.h"

void UpixMulLayer::forward_float()
{
	assert(_info->isrcs.size() == 2);
	assert(_input.shape.w == _output.shape.w);
	assert(_input.shape.h == _output.shape.h);
	assert(_input.shape.c == _output.shape.c * 2);
	const float* inptr0 = _input.fdata.data();
	const float* inptr1 = inptr0 + _output.shape.whc;
	if (_info->isrcs[1].channel_count == 1) {
		// 第二个输入的通道数是1
		int i, j, idx = 0;
		for (i = 0; i < _output.shape.c; i++) {
			for (j = 0; j < _output.shape.wh; j++, idx++) {
				_output.fdata[idx] = inptr0[idx] * inptr1[j];
			}
		}
	}
	else {
		// 第二个输入的分辨率是1*1
		int i, j, idx = 0;
		for (i = 0; i < _output.shape.c; i++) {
			for (j = 0; j < _output.shape.wh; j++, idx++) {
				_output.fdata[idx] = inptr0[idx] * inptr1[i];
			}
		}
	}
}

void UpixMulLayer::forward_int()
{
	assert(_info->isrcs.size() == 2);
	assert(_input.shape.w == _output.shape.w);
	assert(_input.shape.h == _output.shape.h);
	assert(_input.shape.c == _output.shape.c * 2);
	const int* inptr0 = _input.idata.data();
	const int* inptr1 = inptr0 + _output.shape.whc;
	if (_info->isrcs[1].channel_count == 1) {
		// 第二个输入的通道数是1
		int i, j, idx = 0;
		for (i = 0; i < _output.shape.c; i++) {
			for (j = 0; j < _output.shape.w * _output.shape.h; j++, idx++) {
				_output.idata[idx] = inptr0[idx] * inptr1[j];
			}
		}
	}
	else {
		// 第二个输入的分辨率是1*1
		int i, j, idx = 0;
		for (i = 0; i < _output.shape.c; i++) {
			for (j = 0; j < _output.shape.w * _output.shape.h; j++, idx++) {
				_output.idata[idx] = inptr0[idx] * inptr1[i];
			}
		}
	}

	for (auto& v : _output.idata) {
		v >>= _info->quantInt.mulShift;
	}
}
