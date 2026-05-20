#ifndef __S16_BASE_LAYER_H__
#define __S16_BASE_LAYER_H__

#include "S16CNN_Common.h"

typedef enum enS16LayerType {
	S16Layer_Convolution = 0,
	S16Layer_Maxpool,
	S16Layer_Shuffle,
	S16Layer_Upsample,
	S16Layer_Mul,
	S16Layer_Add,
	S16Layer_GlobalAvg,

	S16Layer_MAX
}S16LayerType;

static const char* layer_type_names[S16Layer_MAX] = {
	"conv",
	"maxp",
	"shuffle",
	"upsample",
	"mul",
	"add",
	"global_avg_pool",
};

typedef enum enActivateType {
	ActivateType_Linear = 0,
	ActivateType_Leaky,
	ActivateType_Relu,
	ActivateType_Sigmoid,
	ActivateType_MAX
}ActivateType;

static const char * activate_names[ActivateType_MAX] = {
	"linear",
	"leaky",
	"relu",
	"sigmoid",
};

#define MAX_ACTIVE_FUNCTION_NAME_LENGTH 64
typedef void (*S16_LayerForward_Func) (void *userdata, const S16Tensor *input, S16Tensor *output);

typedef struct stTensorFrom {
	const S16Tensor* t;
	int offset_channel;
	int channels;
	int idx;
}TensorFrom;
#define MAX_FROM_COUNT 8

typedef struct stS16BaseLayer {
	int idx;
	S16LayerType type;
	TensorFrom froms[MAX_FROM_COUNT];
	int from_count;
	S16Tensor input;
	S16Tensor output;

	S16_LayerForward_Func forward;
	int priv_length;
	void *priv;

	int in_whc;		// 在U31中, 输入需要以whc顺序保存
	int out_whc;	// 在U31中, 输出需要以whc顺序保存
}S16BaseLayer;


#endif