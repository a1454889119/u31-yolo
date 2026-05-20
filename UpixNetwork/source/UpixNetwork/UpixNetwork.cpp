#include <assert.h>
#include <memory.h>
#include <stdlib.h>
#include <filesystem>
#include <map>
#include "UpixLayerInfo.h"
#include "Layers/UpixBaseLayer.h"
#include "UpixNetwork.h"

#pragma warning(disable:4996)

struct stLayerQuant {
	bool quantInput;
	bool quantOutput;
	float maxv;
	float minv;
};

static bool quantInput(const UpixLayerInfo* info)
{
	bool bquant = false;
	if (info->type == UpixLayerType::Convolution) {
		bquant = true;
		if (info->activate == ActivateType::Sigmoid || info->idx == 0) {
			bquant = false;
		}
	}
	return bquant;
}

static bool quantOutput(const UpixLayerInfo* info, const std::vector<UpixBaseLayer*>& notQuantLayers)
{
	bool bquant = false;
	if (info->type == UpixLayerType::Convolution) {
		bquant = true;
		if (info->activate == ActivateType::Sigmoid || info->refcount == 0) {
			bquant = false;
		}
		else {
			// mul层之前的输出不量化
			for (auto layer : notQuantLayers) {
				for (auto& from : layer->info()->isrcs) {
					if (from.layer == info->idx) {
						bquant = false;
					}
				}
			}
		}
	}
	else if (info->type == UpixLayerType::Mul) {
		bquant = true;
	}
	return bquant;
}

static void getMaxMin(const std::vector<float>& data, float& maxv, float& minv)
{
	maxv = -FLT_MAX;
	minv = FLT_MAX;
	for (auto v : data) {
		if (v > maxv) maxv = v;
		if (v < minv) minv = v;
	}
}


class UpixNetworkPrivate {
public:
	std::vector<UpixBaseLayer*> layers;
	std::map<int, UpixBaseLayer*> layerMap;
	std::vector<UpixLayerInfo> infos;
	std::vector<stLayerQuant> quants;
	std::vector<UpixBaseLayer*> notQuantLayers;

	int conv_count = 0;
	~UpixNetworkPrivate()
	{
		release();
	}

	void release() {
		for (auto layer : layers) {
			delete layer;
		}
		quants.clear();
		layers.clear();
		layerMap.clear();
		notQuantLayers.clear();
		infos.clear();
		conv_count = 0;
	}
};

UpixNetwork::UpixNetwork()
{
	d = new UpixNetworkPrivate;
}

UpixNetwork::~UpixNetwork()
{
	delete d;
}

bool UpixNetwork::loadNetwork(const char* csv_file)
{
	d->release();
	if (!std::filesystem::exists(csv_file))
		return false;

	if (UpixLayerInfo::load(csv_file, d->infos) > 0) {
		for (auto& info : d->infos) {
			auto layer = UpixBaseLayer::makeLayer(&info);			
			d->layers.push_back(layer);
			d->layerMap.insert({ info.idx, layer });

			if (info.type == UpixLayerType::Convolution) {
				d->conv_count++;
			}
			if (info.type == UpixLayerType::Mul) {
				d->notQuantLayers.push_back(layer);
			}
		}
		
		return true;
	}
	return false;
}

static void loadFloat(FILE* fp, std::vector<float>& data)
{
	float* fptr = data.data();
	int loaded = 0;
	while (true) {
		if (fscanf(fp, "%g,", fptr) == 1) {
			loaded++;
			fptr++;
		}
		else {
			break;
		}
	}
	assert(loaded == (int)data.size());	
}


bool UpixNetwork::loadWeight(const char* weight_file)
{
	bool ok = false;
	FILE* fp = fopen(weight_file, "rt");
	if (fp) {
		char name[64] = { 0 };
		int idx, loaded_layer_count = 0;
		while (true)
		{
			if (fscanf(fp, "layer%d\n", &idx) == 1 && idx >= 0) {
				auto it = d->layerMap.find(idx);
				if (it == d->layerMap.end() || it->second->info()->type != UpixLayerType::Convolution) {
					break;
				}
				// load kernel
				if (fscanf(fp, "%s\n", name) == 1 && strcmp(name, "kernel") == 0) {
					loadFloat(fp, it->second->kernel().fdata);
					fscanf(fp, "\n");
				}
				else {
					break;
				}

				// load biase
				if (fscanf(fp, "%s\n", name) == 1 && strcmp(name, "bias") == 0) {
					loadFloat(fp, it->second->bias().fdata);
					fscanf(fp, "\n");
				}
				else {
					break;
				}
				loaded_layer_count++;
			}
			else {
				break;
			}
		}
		if (loaded_layer_count == d->conv_count) {
			ok = true;
		}
		fclose(fp);
	}
	return ok;
}


static int getQuantIdx(const std::vector<stInputSource>& isrcs, int c)
{
	int cin = 0;
	for (auto& from : isrcs) {
		if (cin <= c && cin + from.channel_count > c) {
			return from.layer;
		}
		cin += from.channel_count;
	}
	return -1;
}

static stLayerQuant getQuant(UpixNetworkPrivate* d, int layerIdx, int cidx)
{
	auto layer = d->layers[layerIdx];
	int c = 0, prev_idx = -1;
	for (auto& from : layer->info()->isrcs) {
		if (c <= cidx && c + from.channel_count > cidx) {
			prev_idx = from.layer;
			break;
		}
		c += from.channel_count;
	}
	assert(prev_idx >= 0);
	
	auto q = d->quants[prev_idx];
	layer = d->layers[prev_idx];
	while (!q.quantOutput) {
		prev_idx = layer->info()->isrcs[0].layer;
		q = d->quants[layer->info()->isrcs[0].layer];
		layer = d->layers[prev_idx];
	}
	return d->quants[layer->info()->idx];
}

static float getAbsMax(const std::vector<float>& data) {
	float maxv = 0.0f;
	for (auto v : data) {
		if (fabsf(v) > maxv) maxv = fabsf(v);
	}
	return maxv;
}

static int log2i(float v)
{
	return (int)(logf(v) / logf(2.0f));

}

static void convertQuantInt(UpixNetworkPrivate* d)
{
	for (auto layer : d->layers) {
		auto info = layer->info();
		auto& q = d->quants[info->idx];
		info->quantInt.quantFlag = QuantIntFlag::None;
		if (info->type == UpixLayerType::Convolution) {
			auto& kernel = layer->kernel();
			auto& bias = layer->bias();
			float kmax = getAbsMax(kernel.fdata);

			float scale = 32767.0f / kmax;
			int iscale = log2i(scale);
			float fscale = (float(1 << iscale));

			assert(iscale >= 10);
			for (size_t i = 0; i < kernel.fdata.size(); i++) {
				kernel.idata[i] = (int)roundf(kernel.fdata[i] * fscale);
				assert(abs(kernel.idata[i]) < 0x7fff);
			}

			if (!q.quantInput) {
				if (info->idx == 0) {
					fscale *= 255.0f;
					iscale += 8;
				}
				else {
					fscale *= 1024.0f;
					iscale += 10;
				}
			}
			
			for (size_t i = 0; i < bias.fdata.size(); i++) {
				bias.idata[i] = (int)roundf(bias.fdata[i] * fscale);				
			}

			float outScale = 255.0f / (q.maxv - q.minv);
			float outZeroPoint = q.minv;

			assert(iscale > 10);

			info->quantInt.mulShift = iscale - 10;
			info->quantInt.outScale = outScale / (1 << 10);
			info->quantInt.outZeroPoint = outZeroPoint * outScale;
			
			
			if (q.quantOutput) {
				info->quantInt.quantFlag = QuantIntFlag::MulSub;
			}
			else {
				if (info->activate != ActivateType::Sigmoid) {
					info->quantInt.quantFlag = QuantIntFlag::ScaleOnly;
				}
			}

			if (info->ksize > 1) {
				for (auto& from : info->isrcs) {
					from.pad_value_i = (int)from.pad_value_f;
				}
			}
		}
		else if (info->type == UpixLayerType::Mul) {
			info->quantInt.quantFlag = QuantIntFlag::MulSub;
			info->quantInt.mulShift = 10;

			float outScale = 255.0f / (q.maxv - q.minv);
			float outZeroPoint = q.minv;

			info->quantInt.outScale = outScale / (1 << 10);
			info->quantInt.outZeroPoint = outZeroPoint * outScale;
		}
		//printf("layer%d  %d\n", info->idx, info->quantInt.mulShift);

	}
}

void UpixNetwork::quant16()
{
	for (auto layer : d->layers) {
		auto info = layer->info();
		info->quantInt.quantFlag = QuantIntFlag::None;
		if (info->idx == 0) {
			info->quantInt.mulShift = 8;
		}
		else {
			info->quantInt.mulShift = 10;
		}

		if (info->type == UpixLayerType::Convolution) {
			auto& kernel = layer->kernel();
			auto& bias = layer->bias();
			int i;
			for (i = 0; i < (int)kernel.fdata.size(); i++) {
				kernel.idata[i] = (int)floorf(kernel.fdata[i] * 1024.0f);
			}

			if (info->idx == 0) {
				for (i = 0; i < (int)bias.fdata.size(); i++) {
					bias.idata[i] = (int)floorf(bias.fdata[i] * 1024.0f * 256.0f);
				}
			}
			else {
				for (i = 0; i < (int)bias.fdata.size(); i++) {
					bias.idata[i] = (int)floorf(bias.fdata[i] * 1024.0f * 1024.0f);
				}
			}			
		}
	}
}

bool UpixNetwork::loadQuant(const char* maxmin_file)
{
	bool ok = false;
	do
	{
		int idx;
		std::map<int, stLayerQuant> quants;

		d->quants.clear();
		if (d->layers.empty()) break;
		d->quants.resize(d->layers.size());

		FILE* fp = fopen(maxmin_file, "rt");
		if (!fp) break;

		stLayerQuant q;
		while (true)
		{
			if (3 == fscanf(fp, "layer%d maxv:%g minv:%g\n", &idx, &q.maxv, &q.minv)) {
				if (idx < 0 || idx >= (int)d->layers.size())
					break;
				auto it = quants.find(idx);
				if (it != quants.end()) {
					break;
				}
				q.quantInput = quantInput(d->layers[idx]->info());
				q.quantOutput = quantOutput(d->layers[idx]->info(), d->notQuantLayers);
				quants.insert({ idx, q });
			}
			else {
				break;
			}
		}
		fclose(fp);

		if (quants.size() != d->layers.size()) break;

		d->quants.resize(d->layers.size());
		for (auto& pair : quants) {
			d->quants[pair.first] = pair.second;
		}

		// apply to weights
		for (auto layer : d->layers) {
			auto info = layer->info();
			if (!d->quants[info->idx].quantInput)
				continue;
			
			auto& kernel = layer->kernel();
			auto& bias = layer->bias();
			
			int i, j, k, t;
			int ksize = info->ksize * info->ksize;
			int kidx = 0, bidx = 0;

			// set pad value
			int c = 0;
			for (auto& from : info->isrcs) {
				auto q = getQuant(d, info->idx, c);
				c += from.channel_count;
				float scale = 255.0f / (q.maxv - q.minv);

				from.pad_value_f = roundf((0.0f - q.minv) * scale);
			}

			for (i = 0; i < info->groups; i++) {
				for (j = 0; j < info->outc / info->groups; j++, bidx++) {
					float boff = 0.0f;
					for (k = 0; k < info->inc / info->groups; k++, kidx += ksize) {
						int c = i * info->inc / info->groups + k;

						auto q = getQuant(d, info->idx, c);
						float scale = (q.maxv - q.minv) / 255.0f;

						for (t = 0; t < ksize; t++) {
							boff += q.minv * kernel.fdata[kidx + t];
							kernel.fdata[kidx + t] *= scale;
						}
					}
					bias.fdata[bidx] += boff;
				}
			}
		}

		convertQuantInt(d);
		ok = true;
	} while (0);

	if (!ok) {
		d->quants.clear();
	}
	return ok;
}

TensorShape UpixNetwork::inputShape() const
{
	TensorShape shape = TENSOR_SHAPE(0, 0, 0);
	if (!d->layers.empty()) {
		shape = d->layers[0]->input();
		shape.fptr = nullptr;
		shape.iptr = nullptr;
	}
	return shape;
}

const std::vector<UpixBaseLayer*>& UpixNetwork::layers() const
{
	return d->layers;
}

TensorShape UpixNetwork::output(int idx) const
{
	TensorShape shape = TENSOR_SHAPE(0, 0, 0);
	auto it = d->layerMap.find(idx);
	if (it != d->layerMap.end()) {
		shape = it->second->output();
	}
	return shape;
}

void quant_output(float* fptr, int count, float maxv, float minv)
{
	assert(maxv > minv);
	float scale = 255.0f / (maxv - minv);
	for (int i = 0; i < count; i++) {
		//fptr[i] = (fptr[i] - minv) * scale;
		fptr[i] = roundf((fptr[i] - minv) * scale);
	}
}

void UpixNetwork::forward(const float* input_data)
{
	if (d->layers.empty()) return;

	auto shape = d->layers[0]->input();
	memcpy(shape.fptr, input_data, sizeof(float) * shape.whc);

	if (d->quants.empty()) {
		for (auto layer : d->layers) {
			auto info = layer->info();
			auto dstShape = layer->input();
			float* dstptr = dstShape.fptr;
			for (auto& src : info->isrcs) {
				auto prevLayer = d->layers[src.layer];
				auto srcShape = prevLayer->output();
				float* srcptr = srcShape.fptr + src.channel_start * srcShape.wh;
				int count = srcShape.wh * src.channel_count;
				memcpy(dstptr, srcptr, sizeof(float) * count);
				dstptr += count;
			}
			layer->forward_float();
		}
	}
	else {
		for (auto layer : d->layers) {
			auto info = layer->info();
			auto dstShape = layer->input();
			float* dstptr = dstShape.fptr;
			for (auto& src : info->isrcs) {
				auto prevLayer = d->layers[src.layer];
				auto srcShape = prevLayer->output();
				float* srcptr = srcShape.fptr + src.channel_start * srcShape.wh;
				int count = srcShape.wh * src.channel_count;
				memcpy(dstptr, srcptr, sizeof(float) * count);
				dstptr += count;
			}
			layer->forward_float();
			if (d->quants[info->idx].quantOutput) {
				auto& quant = d->quants[info->idx];
				quant_output(layer->output().fptr, layer->output().whc, quant.maxv, quant.minv);
			}
		}
	}
}


void UpixNetwork::dumpQuantedWeights(const char* filename)
{
	FILE* fp = nullptr;
	fopen_s(&fp, filename, "wt");
	assert(fp);

	float maxv, minv;
	for (auto layer : d->layers) {
		auto info = layer->info();
		if (info->type == UpixLayerType::Convolution) {
			fprintf_s(fp, "layer%d weights\n", info->idx);
			auto& kernel = layer->kernel();
			auto& bias = layer->bias();

			getMaxMin(kernel.fdata, maxv, minv);
			fprintf_s(fp, "kernel(max:%g, min:%g)\n", maxv, minv);
			for (auto v : kernel.fdata) {
				fprintf_s(fp, "%g, ", v);
			}
			fprintf_s(fp, "\n");

			getMaxMin(bias.fdata, maxv, minv);
			fprintf_s(fp, "bias(max:%g, min:%g)\n", maxv, minv);
			for (auto v : bias.fdata) {
				fprintf_s(fp, "%g, ", v);
			}
			fprintf_s(fp, "\n");
			if (info->ksize > 1) {
				fprintf_s(fp, "pad value\n");
				int c = 0;
				for (auto& from : info->isrcs) {
					fprintf_s(fp, "\tc(%d-%d):%g\n", c, c + from.channel_count, from.pad_value_f);
					c += from.channel_count;
				}
			}			
		}

		auto& q = d->quants[info->idx];
		if (q.quantOutput) {
			fprintf_s(fp, "layer%d quant:", info->idx);
			float scale = 255.0f / (q.maxv - q.minv);

			fprintf_s(fp, "scale:%g, zero_point:%g\n", scale, q.minv);
		}
		fprintf_s(fp, "\n");
	}
	fclose(fp);
}

void quant_output_int(int* iptr, int count, const stQuantInt& quantInt)
{
	for (int i = 0; i < count; i++) {
		float v = iptr[i] * quantInt.outScale - quantInt.outZeroPoint;
		//iptr[i] = roundf(v);
		iptr[i] = (int)(v * 2) - (int)v;
	}
}

void shift_output_int(int* iptr, int count, const stQuantInt& quantInt)
{
	for (int i = 0; i < count; i++) {
		iptr[i] = iptr[i] >> (quantInt.mulShift - 10);
	}
}


static void compare_float_int(TensorShape ts)
{
	int maxdiff = 0, sumdiff = 0;
	float fv = 0.0f;
	int iv = 0;
	for (int i = 0; i < ts.whc; i++) {
		int diff = abs((int)ts.fptr[i] - ts.iptr[i]);
		sumdiff += diff;
		if (diff > maxdiff) {
			maxdiff = diff;
			fv = ts.fptr[i];
			iv = ts.iptr[i];
		}
	}
	printf("maxdiff:%2d, avgdiff:%.5f, %.2f, %3d\n", maxdiff, (float)sumdiff / ts.whc, fv, iv);
}

static void compare_float_int_scale(TensorShape ts, float scale)
{
	float maxdiff = 0, sumdiff = 0;
	float fv = 0.0f;
	int iv = 0;
	for (int i = 0; i < ts.whc; i++) {
		float diff = fabsf(ts.fptr[i] - (float)ts.iptr[i] * scale);
		sumdiff += diff;
		if (diff > maxdiff) {
			maxdiff = diff;
			fv = ts.fptr[i];
			iv = ts.iptr[i];
		}
	}
	printf("maxdiff:%g, avgdiff:%g, %.2f, %3d\n", maxdiff, (float)sumdiff / ts.whc, fv, iv);
}



void UpixNetwork::forward(const int* input_data)
{
	if (d->layers.empty()) return;

	auto shape = d->layers[0]->input();
	memcpy(shape.iptr, input_data, sizeof(int) * shape.whc);

	for (auto layer : d->layers) {
		auto info = layer->info();
		auto dstShape = layer->input();
		int* dstptr = dstShape.iptr;
		for (auto& src : info->isrcs) {
			auto prevLayer = d->layers[src.layer];
			auto srcShape = prevLayer->output();
			int* srcptr = srcShape.iptr + src.channel_start * srcShape.wh;
			int count = srcShape.wh * src.channel_count;
			memcpy(dstptr, srcptr, sizeof(int) * count);
			dstptr += count;
		}
		layer->forward_int();
		switch (info->quantInt.quantFlag) {
		case QuantIntFlag::MulSub:
			quant_output_int(layer->output().iptr, layer->output().whc, info->quantInt);
			//printf("layer%2d:", info->idx);
			//compare_float_int(layer->output());
			break;

		default:
			//printf("layer%2d:", info->idx);
			//compare_float_int_scale(layer->output(), 1.0f / 1024.0f);
			break;
		}
		
	}
}
