#include "UpixBaseLayer.h"

UpixBaseLayer* UpixBaseLayer::makeLayer(UpixLayerInfo* info)
{
	UpixBaseLayer* layer = nullptr;
	switch (info->type) {
	case UpixLayerType::Add:
		info->inc *= (int)info->isrcs.size();
		layer = new UpixAddLayer(info);
		break;

	case UpixLayerType::Convolution:
		layer = new UpixConvLayer(info);
		break;

	case UpixLayerType::GlobalAvg:
		layer = new UpixGAvgLayer(info);
		break;

	case UpixLayerType::Maxpool:
		layer = new UpixMaxpLayer(info);
		break;

	case UpixLayerType::Mul:
		info->inc *= 2;
		layer = new UpixMulLayer(info);
		break;

	case UpixLayerType::Shuffle:
		layer = new UpixShuffleLayer(info);
		break;

	case UpixLayerType::Upsample:
		layer = new UpixUpsampleLayer(info);
		break;

	case UpixLayerType::ReduceMax:		
		layer = new UpixReduceMaxLayer(info);
		break;

	case UpixLayerType::ReduceMean:
		layer = new UpixReduceMeanLayer(info);
		break;

	}
	return layer;
}
