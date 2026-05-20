#ifndef __UPIX_BASE_LAYER_H__
#define __UPIX_BASE_LAYER_H__


#include <UpixNetwork/UpixNetworkCommon.h>
#include <UpixNetwork/UpixLayerinfo.h>

class UpixBaseLayer
{
protected:
	UpixBaseLayer(UpixLayerInfo* info) 
	{
		_info = info; 
		MAKE_TENSOR(info->inw, info->inh, info->inc, _input);
		MAKE_TENSOR(info->outw, info->outh, info->outc, _output);
	}
public:
	static UpixBaseLayer* makeLayer(UpixLayerInfo* info);

	virtual ~UpixBaseLayer() { }

	TensorShape input()const { return _input.shape; }
	TensorShape output()const { return _output.shape; }

	const UpixLayerInfo* info()const { return _info; }
	UpixLayerInfo* info() { return _info; }
	Tensor& kernel() { return _kernel; }
	Tensor& bias() { return _bias; }

	virtual void forward_float() = 0;
	virtual void forward_int() = 0;
	
protected:

	UpixLayerInfo* _info;
	Tensor _input;
	Tensor _output;

	Tensor _kernel;
	Tensor _bias;
};

class UpixAddLayer : public UpixBaseLayer {
public:
	UpixAddLayer(UpixLayerInfo* info) : UpixBaseLayer(info) {};
	virtual void forward_float();
	virtual void forward_int();
};

class UpixConvLayer : public UpixBaseLayer {
public:
	UpixConvLayer(UpixLayerInfo* info) : UpixBaseLayer(info) 
	{
		MAKE_TENSOR(1, 1, info->outc, _bias);
		MAKE_TENSOR(info->ksize, info->ksize, info->inc * info->outc / info->groups, _kernel);
	};
	virtual void forward_float();
	virtual void forward_int();
};

class UpixGAvgLayer : public UpixBaseLayer {
public:
	UpixGAvgLayer(UpixLayerInfo* info) : UpixBaseLayer(info) {};
	virtual void forward_float();
	virtual void forward_int();
};

class UpixMaxpLayer : public UpixBaseLayer {
public:
	UpixMaxpLayer(UpixLayerInfo* info) : UpixBaseLayer(info) {};
	virtual void forward_float();
	virtual void forward_int();
};

class UpixMulLayer : public UpixBaseLayer {
public:
	UpixMulLayer(UpixLayerInfo* info) : UpixBaseLayer(info) {};
	virtual void forward_float();
	virtual void forward_int();
};

class UpixShuffleLayer : public UpixBaseLayer {
public:
	UpixShuffleLayer(UpixLayerInfo* info) : UpixBaseLayer(info) {};
	virtual void forward_float();
	virtual void forward_int();
};

class UpixUpsampleLayer : public UpixBaseLayer {
public:
	UpixUpsampleLayer(UpixLayerInfo* info) : UpixBaseLayer(info) {};
	virtual void forward_float();
	virtual void forward_int();
};

class UpixReduceMaxLayer : public UpixBaseLayer {
public:
	UpixReduceMaxLayer(UpixLayerInfo* info) : UpixBaseLayer(info) {};
	virtual void forward_float();
	virtual void forward_int();
};

class UpixReduceMeanLayer : public UpixBaseLayer {
public:
	UpixReduceMeanLayer(UpixLayerInfo* info) : UpixBaseLayer(info) {};
	virtual void forward_float();
	virtual void forward_int();
};

#endif // !__UPIX_BASE_LAYER_H__
