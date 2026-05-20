#include <stdio.h>
#include <vector>
#include <S16Network/S16Network.h>
#include <S16Network/layers/S16BaseLayer.h>
#include <S16Network/layers/S16ConvLayer.h>
#include "util_func.hpp"

#pragma warning(disable:4996)

static void loadFloat(FILE* fp, std::vector<float>& res)
{
	res.clear();
	while (true) {
		float fv;
		if (fscanf(fp, "%g,", &fv) == 1) {
			res.push_back(fv);
		}
		else {
			break;
		}
	}
}

struct stLayerScale {
	//int rshift = 0;
	int flag = 0;
	int fscale = 1;
	int fzero_point = 0;
	int c_start = 0;
	int c_end = 0;
};

struct stLayerWeight {
	int inc = 0;
	int outc = 0;
	int ksize = 0;
	int group = 0;
	std::vector<float> kernels;
	std::vector<float> biases;
	ActivateType activate;
};

struct stQuantInfo {
	int idx = -1;

	S16LayerType type;
	stLayerWeight weight;

	stLayerScale outputScale;
	std::vector<stLayerScale> inputScales;
};

static bool check_layer_not_quant(const stQuantInfo& layer)
{
	bool notQuant = false;
	if (layer.type == S16Layer_Convolution) {
		if (layer.weight.activate == ActivateType_Sigmoid) {
			notQuant = true;
		}
		else {
			notQuant = false;
		}
	}
	else if (layer.type == S16Layer_Mul) {
		notQuant = false;
	}
	else {
		notQuant = true;
	}
	return notQuant;
}

static void load_weights(const char* in_weights, std::vector<stQuantInfo>& layerInfos)
{
	FILE* fp = fopen(in_weights, "rt");
	if (fp) {		
		char name[64] = { 0 };
		int idx;		
		while (true)
		{
			if (fscanf(fp, "layer%d\n", &idx) == 1 && idx >= 0) {
				auto& w = layerInfos[idx].weight;

				// load kernel
				if (fscanf(fp, "%s\n", name) == 1 && strcmp(name, "kernel") == 0) {
					loadFloat(fp, w.kernels);
				}

				// load biase
				if (fscanf(fp, "%s\n", name) == 1 && strcmp(name, "bias") == 0) {
					loadFloat(fp, w.biases);
				}
			}
			else {
				break;
			}
		}
		fclose(fp);
	}
}

static std::map<std::string, int> load_name_idx(const char* filename)
{
	FILE* fp = fopen(filename, "rt");
	std::map<std::string, int> nameIdx;
	char name[256];
	int idx;
	while (2 == fscanf(fp, "%s %d\n", name, &idx)) {
		nameIdx.insert({ name, idx });
	}
	return nameIdx;
}

static void load_raw_scales(const char* maxminfile, const std::map<std::string, int>& nameIdx, std::vector<stQuantInfo>& layerInfos)
{
	FILE* fp = fopen(maxminfile, "rt");
	assert(fp);
	char name[256];
	float fmax, fmin;
	while (3 == fscanf(fp, "%s %g %g\n",
		name, &fmax, &fmin))
	{
		auto it = nameIdx.find(name);
		assert(it != nameIdx.end());
		auto& layer = layerInfos[it->second];

		if (check_layer_not_quant(layer)) {
			layer.outputScale.flag = 0;
			layer.outputScale.fscale = 1.0f;
			layer.outputScale.fzero_point = 0.0f;
		}
		else {
			layer.outputScale.flag = 1;
			layer.outputScale.fscale = 255.0f / (fmax - fmin);
			layer.outputScale.fzero_point = fmin;
		}
	}
	fclose(fp);
}

void quantlize_weights(
	S16Network* network,
	const char* in_weights_file,
	const char* in_maxmin_file,
	const char* in_name_idx_file,
	const char* out_weights)
{	
	auto layers = network->layers();
	std::vector<stQuantInfo> layerInfos;
	for (auto layer : layers) {
		stQuantInfo info = { layer->idx, layer->type };
		info.outputScale.c_start = 0;
		info.outputScale.c_end = layer->output.c;
		if (layer->type == S16Layer_Convolution) {
			auto conv = (S16ConvLayer*)(layer->priv);
			auto& w = info.weight;
			w.inc = layer->input.c;
			w.outc = layer->output.c;
			w.ksize = conv->ksize;
			w.group = conv->groups;
			w.activate = conv->activate_type;
		}

		layerInfos.push_back(info);
	}
	auto nameIdx = load_name_idx(in_name_idx_file);
	load_raw_scales(in_maxmin_file, nameIdx, layerInfos);

	for (auto& info : layerInfos) {
		auto layer = layers[info.idx];
		int c = 0;
		for (int i = 0; i < layer->from_count; i++) {
			auto& f = layer->froms[i];
			stLayerScale scale;
			scale.c_start = c;
			scale.c_end = c + f.channels;
			c += f.channels;
			scale.flag = layerInfos[f.idx].outputScale.flag;
			scale.fscale = layerInfos[f.idx].outputScale.fscale;
			scale.fzero_point = layerInfos[f.idx].outputScale.fzero_point;
			info.inputScales.push_back(scale);
		}

		if (info.type == S16Layer_Maxpool || info.type == S16Layer_Upsample) {
			// 对于这种不用做乘加运算的层, 需要把输入的scale放到输出
			assert(info.inputScales.size() == 1);
			info.outputScale.fscale = info.inputScales[0].fscale;
			info.outputScale.fzero_point = info.inputScales[0].fzero_point;
		}
	}
	load_weights(in_weights_file, layerInfos);

	for (auto& info : layerInfos) {
		if (info.type == S16Layer_Convolution) {
			auto& w = info.weight;
			int kcount = w.ksize * w.ksize;
			int kidx = 0, bidx = 0;
			if (info.inputScales.size() == 1) {
				auto& s = info.inputScales[0];
				if (s.flag) {
					for (auto& k : w.kernels) {
						//k *= s.scale * (1.0f / 1024.0f);						
					}

					// 每个偏移量对应n个输入通道
					int n = w.inc / w.group;
					for (auto& b : w.biases) {
						//b += s.zero_point * s.scale * (1.0f / 1024.0f) * n;
					}
				}
			}
			else if (info.inputScales.size() > 1) {

			}
		}
	}
}