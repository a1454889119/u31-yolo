#ifndef __UPIX_NETWORK_COMMON_H__
#define __UPIX_NETWORK_COMMON_H__

#include <vector>

typedef struct stTensorShape {
	int w = 0;
	int h = 0;
	int c = 0;
	int wh = 0;
	int whc = 0;

	float* fptr = nullptr;
	int* iptr = nullptr;
}TensorShape;
#define TENSOR_SHAPE(w, h, c) {w, h, c, w*h, w*h*c, nullptr, nullptr}

typedef struct stTensor {
	TensorShape shape;
	std::vector<float> fdata;
	std::vector<int> idata;
}Tensor;

#define MAKE_TENSOR(w, h, c, t) \
	t.shape = TENSOR_SHAPE(w, h, c);\
	t.fdata.resize(t.shape.whc);\
	t.idata.resize(t.shape.whc);\
	t.shape.fptr = t.fdata.data();\
	t.shape.iptr = t.idata.data();

enum class UpixLayerType
{
	Convolution = 0,
	Maxpool,
	Shuffle,
	Upsample,
	Mul,
	Add,
	GlobalAvg,
	ReduceMax,
	ReduceMean,

	MAX
};

enum class ActivateType
{
	Linear = 0,
	Leaky,
	Relu,
	Sigmoid,
	MAX
};

struct stInputSource {
	int layer = -1;
	int channel_start = 0;
	int channel_count = -1;

	float pad_value_f = 0.0f;
	int pad_value_i = 0;
};

#endif // !__UPIX_NETWORK_COMMON_H__
