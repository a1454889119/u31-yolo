#include <assert.h>
#include <fstream>
#include <string>
#include <memory.h>
#include "layers/S16CNN_Common.h"
#include "layers/S16BaseLayer.h"
#include "layers/S16ConvLayer.h"
#include "layers/S16MaxpoolLayer.h"
#include "layers/S16ShuffleLayer.h"
#include "layers/S16UpsampleLayer.h"
#include "layers/S16Add.h"
#include "layers/S16Mul.h"
#include "layers/S16GlobalAvg.h"

#include "S16Network.h"

typedef std::vector<std::string> StringArray;
static std::string trim(const std::string& text, const std::string& chars)
{
	std::string newText;
	size_t start = 0, length = text.length(), end = length - 1;
	for (size_t i = 0; i < length; i++)
	{
		int n = (int)chars.find_first_of(text[i]);
		if (n >= 0) {
			start++;
		}
		else
			break;
	}

	for (int i = (int)length - 1; i >= 0; i--) {
		int n = (int)chars.find_first_of(text[i]);
		if (n >= 0) {
			end--;
		}
		else
			break;
	}

	return text.substr(start, end - start + 1);
}

static StringArray splitByChars(const std::string& text, const std::string& chars)
{
	StringArray parts;
	std::string t;
	for (auto c : text) {
		int n = (int)chars.find_first_of(c);
		if (n >= 0) {
			parts.push_back(t);
			t.clear();
		}
		else {
			t += c;
		}
	}
	if (t.length() > 0) {
		parts.push_back(t);
	}
	return parts;
}

bool parseDecimal(const std::string& text, int* v)
{
	int len = (int)text.size();
	if (len == 0 || len > 10) return false;
	if (len > 1 && text[0] == ('0')) return false;
	uint64_t value = 0;
	for (int i = 0; i < len; i++) {
		int n = text[i] - ('0');
		if (n < 0 || n >= 10) return false;
		value += n;
		if (i != len - 1)
			value *= 10;
	}

	if (value > INT_MAX)
		return false;
	*v = (int)value;
	return true;
}


static std::vector<StringArray> loadCSV(const std::string filename)
{
	std::vector<StringArray> result;
	std::ifstream ifs(filename);

	while (!ifs.eof()) {
		std::string line;
		getline(ifs, line);
		auto text = trim(line, "\r\n\t\" ");
		auto parts = splitByChars(text, ",");
		StringArray trimed_parts;
		for (auto text : parts) {
			trimed_parts.push_back(trim(text, "\r\n\t\" "));
		}
		result.push_back(trimed_parts);
	}
	ifs.close();
	return result;
}

struct stFromInfo {
	int layer;
	int channel_start = 0;
	int channel_count = -1;
};

struct stLayerInfo {
	int idx;
	int inw;
	int inh;
	int inc;
	int outw;
	int outh;
	int outc;
	S16LayerType type = S16Layer_MAX;
	ActivateType activate = ActivateType_Linear;
	int ksize = -1;
	int stride = -1;
	int pad = -1;
	int groups = 1;
	std::vector<stFromInfo> from;
};

static stLayerInfo decodeLine(const StringArray& line)
{
	stLayerInfo info;
	int i;
	if (line.size() != 10)
		throw std::string("Need 10 parts in a line");

	const std::string idx_name = line[0];
	const std::string io_name = line[1];
	const std::string type_name = line[2];
	const std::string act_name = line[3];
	const std::string ksize_name = line[4];
	const std::string stride_name = line[5];
	const std::string pad_name = line[6];
	const std::string groups_name = line[7];
	const std::string from_name = line[8];
	const std::string offset_name = line[9];
	if (!parseDecimal(idx_name, &info.idx))
		throw "Wrong idx '" + idx_name + "'";
	
	if (6 != sscanf(io_name.c_str(), "%d*%d*%d->%d*%d*%d",
		&info.inw, &info.inh, &info.inc,
		&info.outw, &info.outh, &info.outc))
		throw "Wrong in->out '" + io_name + "'";

	for (i = 0; i < (int)S16Layer_MAX; i++) {
		if (type_name == layer_type_names[i]) {
			info.type = (S16LayerType)i;
			break;
		}
	}
	if (i == (int)S16Layer_MAX) {
		throw "Unknown type name '" + type_name + "'";
	}

	if (!act_name.empty()) {

		for (i = 0; i < (int)ActivateType_MAX; i++) {
			if (act_name == activate_names[i]) {
				info.activate = (ActivateType)i;
				break;
			}
		}
		if (i == (int)ActivateType_MAX) {
			throw "Unknown activate name '" + act_name + "'";
		}
	}

	if (!ksize_name.empty()) {
		if (!parseDecimal(ksize_name, &info.ksize))
			throw "Wrong ksize '" + ksize_name + "'";
	}

	if (!stride_name.empty()) {
		if (!parseDecimal(stride_name, &info.stride))
			throw "Wrong stride '" + stride_name + "'";
	}

	if (!pad_name.empty()) {
		if (!parseDecimal(pad_name, &info.pad))
			throw "Wrong pad '" + pad_name + "'";
	}

	if (!groups_name.empty()) {
		if (!parseDecimal(groups_name, &info.groups))
			throw "Wrong pad '" + groups_name + "'";
	}

	if (!from_name.empty()) {
		auto parts = splitByChars(from_name, " ");
		for (auto& p : parts) {
			int idx;
			if (!parseDecimal(p, &idx)) {
				throw "Bad number in 'from' : '" + p + "'";
			}

			info.from.push_back({ idx });
		}

		if (offset_name.empty()) {
			throw "Bad content in 'offset channel' : '" + offset_name + "'";
		}
	}

	if (!offset_name.empty()) {
		auto parts = splitByChars(offset_name, " ");
		if (parts.size() != info.from.size()) {
			throw "Bad content in 'offset channel' : '" + offset_name + "'";
		}
		for (size_t i=0; i<parts.size(); i++) {
			int n1, n2;
			if (parseDecimal(parts[i], &n1)) {
				info.from[i].channel_start = n1;
			}
			else {
				auto pp = splitByChars(parts[i], "+");
				if (pp.size() == 2) {
					if (parseDecimal(pp[0], &n1) && parseDecimal(pp[1], &n2)) {
						info.from[i].channel_start = n1;
						info.from[i].channel_count = n2;
					}
					else {
						throw "Bad content in 'offset channel' : '" + parts[i] + "'";
					}
				}
				else {
					throw "Bad content in 'offset channel' : '" + parts[i] + "'";
				}
			}
		}
	}
	return info;
}

static stLayerInfo decodeLine_v1(const StringArray& line)
{
	stLayerInfo info;
	int i;
	if (line.size() != 10)
		throw std::string("Need 10 parts in a line");

	const std::string idx_name = line[0];
	const std::string io_name = line[1];
	const std::string type_name = line[2];
	const std::string act_name = line[3];
	const std::string ksize_name = line[4];
	const std::string stride_name = line[5];
	const std::string pad_name = line[6];
	const std::string groups_name = line[7];
	const std::string inputs_name = line[8];

	if (idx_name.size() != 5 || idx_name[0] != '[' || idx_name[4] != ']') {
		throw "bad idx :" + idx_name;
	}
	

	if (sscanf(idx_name.c_str(), "[%3d]", &info.idx) != 1)
		throw "bad idx :" + idx_name;

	if (6 != sscanf(io_name.c_str(), "%3d*%3d*%3d->%3d*%3d*%3d",
		&info.inw, &info.inh, &info.inc,
		&info.outw, &info.outh, &info.outc))
		throw "Wrong in->out '" + io_name + "'";

	for (i = 0; i < (int)S16Layer_MAX; i++) {
		if (type_name == layer_type_names[i]) {
			info.type = (S16LayerType)i;
			break;
		}
	}
	if (i == (int)S16Layer_MAX) {
		throw "Unknown type name '" + type_name + "'";
	}

	if (!act_name.empty()) {

		for (i = 0; i < (int)ActivateType_MAX; i++) {
			if (act_name == activate_names[i]) {
				info.activate = (ActivateType)i;
				break;
			}
		}
		if (i == (int)ActivateType_MAX) {
			throw "Unknown activate name '" + act_name + "'";
		}
	}

	if (!ksize_name.empty()) {
		if (!parseDecimal(ksize_name, &info.ksize))
			throw "Wrong ksize '" + ksize_name + "'";
	}

	if (!stride_name.empty()) {
		if (!parseDecimal(stride_name, &info.stride))
			throw "Wrong stride '" + stride_name + "'";
	}

	if (!pad_name.empty()) {
		if (!parseDecimal(pad_name, &info.pad))
			throw "Wrong pad '" + pad_name + "'";
	}

	if (!groups_name.empty()) {
		if (!parseDecimal(groups_name, &info.groups))
			throw "Wrong pad '" + groups_name + "'";
	}

	if (!inputs_name.empty()) {
		auto parts = splitByChars(inputs_name, ";");
		for (auto& p : parts) {
			stFromInfo from;
			if (3 != sscanf(p.c_str()+3, "[%3d]%2d+%2d", &from.layer, &from.channel_start, &from.channel_count)) {
				throw "Bad content in 'inputs_name' : " + p;
			}
			info.from.push_back(from);
		}
	}
	return info;
}

static S16BaseLayer* makeConv(const stLayerInfo* info)
{
	int in_count = 0;
	if (info->from.size() > 1) {
		in_count = info->inw * info->inh * info->inc;
	}
	int out_count = info->outw * info->outh * info->outc;
	int kcount = info->ksize * info->ksize * info->outc * info->inc / info->groups;
	int bcount = info->outc;
	int total_size = sizeof(S16BaseLayer) + sizeof(S16ConvLayer);
	total_size += sizeof(s16) * (kcount + out_count + in_count);
	total_size += sizeof(int)* bcount;
	S16BaseLayer* layer = (S16BaseLayer*)malloc(total_size);
	if (!layer)
		return nullptr;
		
	S16ConvLayer* conv = (S16ConvLayer*)(layer + 1);
	layer->priv = conv;
	layer->in_whc = 0;
	layer->out_whc = 0;
	conv->groups = info->groups;
	conv->ksize = info->ksize;
	conv->stride = info->stride;
	conv->pad = info->pad;
	conv->kernels = (s16*)(conv + 1);
	conv->biases = (int*)(conv->kernels + kcount);
	conv->activate_type = info->activate;
	conv->shift = S16_QUANT_SHIFT;

	layer->forward = S16_ConvolutionForward;
	layer->output.ptr = (s16*)(conv->biases + bcount);
	layer->input.ptr = nullptr;
	layer->output.quant_count = 0;
	if (in_count) {
		layer->input.ptr = layer->output.ptr + out_count;
	}

	if (conv->groups == info->outc) {
		layer->in_whc = 1;
	}
	return layer;
}

static S16BaseLayer* makeMaxp(const stLayerInfo* info)
{
	int in_count = 0;
	if (info->from.size() > 1) {
		in_count = info->inw * info->inh * info->inc;
	}
	int out_count = info->outw * info->outh * info->outc;
	int total_size = sizeof(S16BaseLayer) + sizeof(S16MaxpoolLayer) + sizeof(s16) * (out_count + in_count);
	S16BaseLayer* layer = (S16BaseLayer*)malloc(total_size);
	if (!layer)
		return nullptr;
	S16MaxpoolLayer* maxp = (S16MaxpoolLayer*)(layer + 1);
	layer->priv = maxp;
	layer->out_whc = 0;

	maxp->size = info->ksize;
	maxp->stride = info->stride;
	maxp->pad = info->pad;

	layer->forward = S16_MaxpoolForward;
	layer->output.ptr = (s16*)(maxp + 1);
	layer->input.ptr = nullptr;
	if (in_count) {
		layer->input.ptr = layer->output.ptr + out_count;
	}
	layer->in_whc = 1;
	return layer;
}

static S16BaseLayer* makeShuffle(const stLayerInfo* info)
{
	int in_count = 0;
	if (info->from.size() > 1) {
		in_count = info->inw * info->inh * info->inc;
	}
	int out_count = info->outw * info->outh * info->outc;
	int total_size = sizeof(S16BaseLayer) + sizeof(S16ShuffleLayer) + sizeof(s16) * (out_count + in_count);
	S16BaseLayer* layer = (S16BaseLayer*)malloc(total_size);

	if (!layer)
		return nullptr;
	S16ShuffleLayer* shuffle = (S16ShuffleLayer*)(layer + 1);
	layer->priv = shuffle;
	layer->in_whc = 0;
	layer->out_whc = 0;

	layer->forward = S16_ShuffleForward;
	layer->output.ptr = (s16*)(shuffle + 1);
	layer->input.ptr = nullptr;
	if (in_count) {
		layer->input.ptr = layer->output.ptr + out_count;
	}
	return layer;
}

static S16BaseLayer* makeUpsample(const stLayerInfo* info)
{
	int in_count = 0;
	if (info->from.size() > 1) {
		in_count = info->inw * info->inh * info->inc;
	}
	int out_count = info->outw * info->outh * info->outc;
	int total_size = sizeof(S16BaseLayer) + sizeof(S16UpsampleLayer) + sizeof(s16) * (out_count + in_count);
	S16BaseLayer* layer = (S16BaseLayer*)malloc(total_size);

	if (!layer)
		return nullptr;
	S16UpsampleLayer* upsample = (S16UpsampleLayer*)(layer + 1);
	layer->priv = upsample;
	layer->in_whc = 1;
	layer->out_whc = 0;

	upsample->ksize = info->ksize;
	layer->forward = S16_UpsampleForward;
	layer->output.ptr = (s16*)(upsample + 1);
	layer->input.ptr = nullptr;
	if (in_count) {
		layer->input.ptr = layer->output.ptr + out_count;
	}
	return layer;
}

static S16BaseLayer* makeMul(const stLayerInfo* info)
{
	int in_count = 0;
	if (info->from.size() > 1) {
		in_count = info->outw * info->outh * info->outc + info->outc;
	}
	int out_count = info->outw * info->outh * info->outc;
	int total_size = sizeof(S16BaseLayer) + sizeof(S16MulLayer) + sizeof(s16) * (out_count + in_count);
	S16BaseLayer* layer = (S16BaseLayer*)malloc(total_size);

	if (!layer)
		return nullptr;
	S16MulLayer* mul = (S16MulLayer*)(layer + 1);
	layer->priv = (S16MulLayer*)(layer + 1);
	layer->in_whc = 1;
	layer->out_whc = 0;

	layer->forward = S16_MulForward;
	layer->output.ptr = (s16*)(mul + 1);
	layer->input.ptr = nullptr;
	if (in_count) {
		layer->input.ptr = layer->output.ptr + out_count;
	}
	return layer;
}

static S16BaseLayer* makeAdd(const stLayerInfo* info)
{
	int in_count = 0;
	if (info->from.size() > 1) {
		in_count = info->inw * info->inh * info->inc;
	}
	int out_count = info->outw * info->outh * info->outc;
	int total_size = sizeof(S16BaseLayer) + sizeof(S16AddLayer) + sizeof(s16) * (out_count + in_count);
	S16BaseLayer* layer = (S16BaseLayer*)malloc(total_size);

	if (!layer)
		return nullptr;
	S16AddLayer* add = (S16AddLayer*)(layer + 1);
	layer->priv = add;
	layer->in_whc = 1;
	layer->out_whc = 0;

	add->activate_type = info->activate;
	layer->forward = S16_AddForward;
	layer->output.ptr = (s16*)(add + 1);
	layer->input.ptr = nullptr;
	if (in_count) {
		layer->input.ptr = layer->output.ptr + out_count;
	}
	return layer;
}

static S16BaseLayer* makeGlobalAvg(const stLayerInfo* info)
{
	int in_count = 0;
	if (info->from.size() > 1) {
		in_count = info->inw * info->inh * info->inc;
	}
	int out_count = info->outw * info->outh * info->outc;
	int total_size = sizeof(S16BaseLayer) + sizeof(S16GlobalAvgLayer) + sizeof(s16) * (out_count + in_count);
	S16BaseLayer* layer = (S16BaseLayer*)malloc(total_size);

	if (!layer)
		return nullptr;
	S16GlobalAvgLayer* gal = (S16GlobalAvgLayer*)(layer + 1);
	layer->priv = gal;
	layer->in_whc = 1;
	layer->out_whc = 0;

	layer->forward = S16_GlobalAvgForward;
	layer->output.ptr = (s16*)(gal + 1);
	layer->input.ptr = nullptr;
	if (in_count) {
		layer->input.ptr = layer->output.ptr + out_count;
	}
	return layer;
}

static S16BaseLayer* create_layer_by_info(const stLayerInfo* info)
{
	S16BaseLayer* layer = nullptr;
	switch (info->type) {
	case S16Layer_Convolution: layer = makeConv(info); break;
	case S16Layer_Maxpool: layer = makeMaxp(info); break;
	case S16Layer_Shuffle: layer = makeShuffle(info); break;
	case S16Layer_Upsample: layer = makeUpsample(info); break;
	case S16Layer_Mul: layer = makeMul(info); break;
	case S16Layer_Add: layer = makeAdd(info); break;
	case S16Layer_GlobalAvg: layer = makeGlobalAvg(info); break;
	default: throw "Bad layer type for idx:" + std::to_string(info->idx);
	}
	layer->idx = info->idx;
	layer->type = info->type;
	layer->input.w = info->inw;
	layer->input.h = info->inh;
	layer->input.c = info->inc;
	layer->input.whc = info->inw * info->inh * info->inc;

	layer->output.w = info->outw;
	layer->output.h = info->outh;
	layer->output.c = info->outc;
	layer->output.whc = info->outw * info->outh * info->outc;
	return layer;
}

#ifndef max
#define max(a, b) (a) > (b) ? (a) : (b)
#endif // !max

#ifndef min
#define min(a, b) (a) < (b) ? (a) : (b)
#endif // !max


class S16NetworkPrivate
{
public:
	std::vector<S16BaseLayer*> layers;
	std::map<int, S16BaseLayer*> layerMap;
	float* fdata = nullptr;
	int fcount = 0;
	int conv_count = 0;
	void release() 
	{
		for (auto layer : layers) {
			free(layer);
		}
		layers.clear();
		layerMap.clear();

		if (fdata) free(fdata);
		fdata = nullptr;
		fcount = 0;
		conv_count = 0;
	}

	~S16NetworkPrivate()
	{
		release();
	}

	void add_layer_by_info(const stLayerInfo* info)
	{
		S16BaseLayer* layer = create_layer_by_info(info);
		std::vector<S16BaseLayer*> inputLayers;
		if (layer->input.ptr == nullptr) {
			if (info->from.size() == 1) {
				auto it = layerMap.find(info->from[0].layer);
				if (it == layerMap.end()) {
					throw "Bad from(" + std::to_string(info->from[0].layer) + ") in layer" + std::to_string(info->idx);
				}

				layer->froms[0].t = &it->second->output;
				layer->froms[0].offset_channel = info->from[0].channel_start;
				layer->froms[0].channels = info->from[0].channel_count;
				layer->froms[0].idx = it->second->idx;
				if (layer->froms[0].channels == -1) {
					layer->froms[0].channels = it->second->output.c;
				}

				layer->input.ptr = it->second->output.ptr;
				if (layer->froms[0].offset_channel > 0) {
					layer->input.ptr += layer->froms[0].offset_channel * it->second->output.w * it->second->output.h;
				}
				inputLayers.push_back(it->second);
			}
		}
		else {
			int c = 0;
			for (int i = 0; i < (int)info->from.size(); i++) {
				auto it = layerMap.find(info->from[i].layer);
				if (it == layerMap.end()) {
					throw "Bad from(" + std::to_string(info->from[i].layer) + ") in layer" + std::to_string(info->idx);
				}
				layer->froms[i].t = &it->second->output;
				layer->froms[i].offset_channel = info->from[i].channel_start;
				layer->froms[i].channels = info->from[i].channel_count;
				layer->froms[i].idx = it->second->idx;
				if (layer->froms[i].channels == -1) {
					layer->froms[i].channels = it->second->output.c;
				}
				inputLayers.push_back(it->second);
				c += layer->froms[i].channels;
			}
			if (layer->input.c != c) {
				if (layer->type == S16Layer_Mul) {
					layer->input.c = c;
				}
				else {
					assert(0);
				}
			}			
		}
		layer->from_count = (int)info->from.size();
		if (layer->in_whc) {
			for (auto prev : inputLayers) {
				prev->out_whc = 1;
			}
		}

		if (layer->type == S16Layer_Convolution) {
			conv_count++;
			S16ConvLayer* conv = (S16ConvLayer*)layer->priv;
			if (layer->idx == 0) {				
				conv->shift = 8;
			}

			int kcount = conv->ksize * conv->ksize * layer->input.c * layer->output.c / conv->groups;
			int bcount = layer->output.c;
			int n = max(kcount, bcount);

			if (n > fcount) {
				fcount = n;
			}
		}
				
		layers.push_back(layer);
		layerMap.insert({ info->idx, layer });
	}
};

S16Network::S16Network()
{
	d = new S16NetworkPrivate;
}

S16Network::~S16Network()
{
	delete d;
}

int S16Network::inputW() const
{
	int w = 0;
	if (d->layers.size()) {
		w = d->layers[0]->input.w;
	}
	return w;
}

int S16Network::inputH() const
{
	int h = 0;
	if (d->layers.size()) {
		h = d->layers[0]->input.h;
	}
	return h;
}

int S16Network::inputC() const
{
	int c = 0;
	if (d->layers.size()) {
		c = d->layers[0]->input.c;
	}
	return c;
}

bool S16Network::loadNetwork(const char* csv_file, int version)
{
	auto lines = loadCSV(csv_file);
	d->release();
	try
	{
		for (auto& line : lines) {
			if (line.size() <= 0) continue;
			if (line.size() == 1 && line[0].empty()) continue;

			if (line[0].size() >= 2 && line[0][0] == '/' && line[0][1] == '/') {
				continue;
			}

			struct stLayerInfo info;
			switch (version) {
			case 0:
				info = decodeLine(line);
				break;

			case 1:
				info = decodeLine_v1(line);
				break;

			default: throw "Unknown Version";
			}
			d->add_layer_by_info(&info);
		}
	}
	catch (const std::string& err)
	{
		printf("%s\n", err.c_str());
		d->release();
		return false;
	}

	d->fdata = (float*)malloc(sizeof(float) * d->fcount);	
	return d->layers.size() > 0;
}


static int loadFloat(FILE* fp, float* fptr)
{
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
	return loaded;
}

bool S16Network::loadWeight(const char* weight_file)
{
	bool ok = false;
	FILE* fp = fopen(weight_file, "rt");
	if (fp) {
		char name[64] = { 0 };
		int idx, i;
		int loaded_layer_count = 0;
		while (true)
		{
			if (fscanf(fp, "layer%d\n", &idx) == 1 && idx >= 0) {
				auto it = d->layerMap.find(idx);
				if (it == d->layerMap.end() || it->second->type != S16Layer_Convolution) {
					break;
				}
				S16ConvLayer* conv = (S16ConvLayer*)it->second->priv;
				int kcount = conv->ksize * conv->ksize * it->second->input.c * it->second->output.c / conv->groups;
				int bcount = it->second->output.c;

				// load kernel
				if (fscanf(fp, "%s\n", name) == 1 && strcmp(name, "kernel") == 0) {
					if (loadFloat(fp, d->fdata) != kcount) {
						break;
					}
					fscanf(fp, "\n");

					s16* sptr = conv->kernels;
					for (i = 0; i < kcount; i++) {
						sptr[i] = (s16)(d->fdata[i] * 1024.0f);
					}
				}
				else {
					break;
				}

				// load biase
				if (fscanf(fp, "%s\n", name) == 1 && strcmp(name, "bias") == 0) {
					if (loadFloat(fp, d->fdata) != bcount) {
						break;
					}
					fscanf(fp, "\n");

					int* sptr = conv->biases;
					if (idx != 0) {
						for (i = 0; i < bcount; i++) {
							sptr[i] = (int)(d->fdata[i] * 1024.0f * 1024.0f);
						}
					}
					else {
						for (i = 0; i < bcount; i++) {
							sptr[i] = (int)(d->fdata[i] * 1024.0f * 256.0f);
						}
					}
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

void S16Network::forward(const short* input_data)
{
	d->layers[0]->input.ptr = (short*)input_data;
	for (auto layer : d->layers) {

		if (layer->from_count > 1)
		{
			short* dst = layer->input.ptr;
			for (int i = 0; i < layer->from_count; i++) {
				const S16Tensor* t = layer->froms[i].t;
				int wh = t->w * t->h;
				int whc = wh * layer->froms[i].channels;
				const short* src = t->ptr;
				src += layer->froms[i].offset_channel * wh;
				memcpy(dst, src, sizeof(s16) * whc);

				dst += whc;
			}
		}
		//printf("layer%d forward\n", layer->idx);
		layer->forward(layer->priv, &layer->input, &layer->output);
	}
}

void S16Network::forward_quant8bit(const short* input_data)
{
	d->layers[0]->input.quants[0] = { 1, 0, 0, d->layers[0]->input.c  };
	d->layers[0]->input.quant_count = 1;
	d->layers[0]->input.ptr = (short*)input_data;

	int maxwhc = 0;
	for (auto layer : d->layers) {
		if (layer->input.whc > maxwhc) {
			maxwhc = layer->input.whc;
		}
	}
	short* layer_input_data = (short*)malloc(sizeof(short) * maxwhc);
	assert(layer_input_data);

	for (auto layer : d->layers) {
		if (layer->from_count >= 1)
		{
			short* dst = layer_input_data;
			int c = 0;
			for (int i = 0; i < layer->from_count; i++) {
				const S16Tensor* t = layer->froms[i].t;
				int wh = t->w * t->h;
				int whc = wh * layer->froms[i].channels;
				const short* src = t->ptr;
				src += layer->froms[i].offset_channel * wh;
				memcpy(dst, src, sizeof(s16) * whc);

				dst += whc;
				layer->input.quants[i].scale = t->quants[0].scale;
				layer->input.quants[i].zero_point = t->quants[0].zero_point;
				layer->input.quants[i].c_start = c;
				layer->input.quants[i].c_end = c + layer->froms[i].channels;
				c += layer->froms[i].channels;
			}
			layer->input.quant_count = layer->from_count;
			layer->input.ptr = layer_input_data;
		}
		else {
			layer->input.quant_count = 1;
			if (layer->from_count == 1) {
				const S16Tensor* t = layer->froms[0].t;
				layer->input.quants[0] = t->quants[0];
			}
			layer->input.quants[0].c_start = 0;
			layer->input.quants[0].c_end = layer->input.c;
		}


		// de-quantize input
		short* inptr = layer->input.ptr;
		short* input_end = inptr + layer->input.whc;
		int i, wh = layer->input.w * layer->input.h;
		for (i = 0; i < layer->input.quant_count; i++) {
			int c = layer->input.quants[i].c_start;
			int scale = layer->input.quants[i].scale;
			int zero_point = layer->input.quants[i].zero_point;

			short* in_end = inptr + (layer->input.quants[i].c_end - layer->input.quants[i].c_start) * wh;
			while (inptr < in_end && inptr < input_end) {
				*inptr = *inptr * scale + zero_point;
				inptr++;
			}
		}

		layer->forward(layer->priv, &layer->input, &layer->output);

		// quantize output
		if (layer->output.quants[0].scale != 1 && layer->output.quants[0].zero_point != 0) {
			int zero_point = layer->output.quants[0].zero_point;
			int scale = layer->output.quants[0].scale;
			float scale_inv = 1.0f / scale;

			short* outptr = layer->output.ptr;
			short* out_end = outptr + layer->output.whc;
			while (outptr < out_end) {
				int v = *outptr - zero_point;
				float fv = v * scale_inv * 8.0f;
				fv = max(fv, 0.0f);
				fv = min(fv, +2047.0f);

				v = (int)(fv) >> 3;
				*outptr = v;
				outptr++;
			}
		}
	}
	free(layer_input_data);
}

static void dump_tensor_whc(const char* filename, const S16Tensor* t)
{
	FILE* fp = fopen(filename, "wt");
	if (fp) {
		const short* ptr = t->ptr;
		int i;
		for (i = 0; i < t->whc; i++, ptr++) {
			fprintf(fp, "%04X ", (unsigned short)*ptr);
			if ((i + 1) % 16 == 0) fprintf(fp, "\n");
		}

		fclose(fp);
	}
}

static void dump_tensor_cwh(const char* filename, const S16Tensor* t)
{
	FILE* fp = fopen(filename, "wt");
	if (fp) {
		const short* ptr = t->ptr;
		int wh = t->w * t->h;
		int i, j, k = 0;
		for (i = 0; i < wh; i++, ptr++) {
			const short* cptr = ptr;
			for (j = 0; j < t->c; j++, k++, cptr += wh) {
				fprintf(fp, "%04X ", (unsigned short)*cptr);
				if ((k + 1) % 16 == 0) fprintf(fp, "\n");
			}
		}

		fclose(fp);
	}
}

static void dump_tensor_whc_8bit(const char* filename, const S16Tensor* t)
{
	FILE* fp = fopen(filename, "wt");
	if (fp) {
		const short* ptr = t->ptr;
		int i;
		for (i = 0; i < t->whc; i++, ptr++) {
			fprintf(fp, "%02X ", (unsigned char)*ptr);
			if ((i + 1) % 32 == 0) fprintf(fp, "\n");
		}

		fclose(fp);
	}
}

static void dump_tensor_cwh_8bit(const char* filename, const S16Tensor* t)
{	
	FILE* fp = fopen(filename, "wt");
	if (fp) {
		const short* ptr = t->ptr;
		int wh = t->w * t->h;
		int i, j, k = 0;
		for (i = 0; i < wh; i++, ptr++) {
			const short* cptr = ptr;
			for (j = 0; j < t->c; j++, k++, cptr += wh) {
				fprintf(fp, "%02X ", (unsigned char)*cptr);
				if ((k + 1) % 32 == 0) fprintf(fp, "\n");
			}
		}

		fclose(fp);
	}
}

void S16Network::dumpToDir(const char* dir_name, int flag_8bit)const
{
	char filename[128];
	for (auto layer : d->layers) {
		if (flag_8bit && (layer->output.quants[0].scale != 1 && layer->output.quants[0].zero_point != 0)) {
			sprintf(filename, "%s/layer%d-s8.txt", dir_name, layer->idx);
			if (layer->out_whc) {
				dump_tensor_whc_8bit(filename, &layer->output);
			}
			else {
				dump_tensor_cwh_8bit(filename, &layer->output);
			}
		}
		else {
			sprintf(filename, "%s/layer%d-s16.txt", dir_name, layer->idx);
			if (layer->out_whc) {
				dump_tensor_whc(filename, &layer->output);
			}
			else {
				dump_tensor_cwh(filename, &layer->output);
			}
		}
	}
}

const short* S16Network::getLayerOutput(int idx) const
{
	const short* ptr = nullptr;
	auto it = d->layerMap.find(idx);
	if (it != d->layerMap.end()) {
		ptr = it->second->output.ptr;
	}
	return ptr;
}

const std::vector<S16BaseLayer*>& S16Network::layers() const
{
	return d->layers;
}

const std::map<int, S16BaseLayer*>& S16Network::layerMap() const
{
	return d->layerMap;
}

void S16Network::dumpLayer(int idx, const char* filename) const
{
	auto it = d->layerMap.find(idx);
	if (it == d->layerMap.end()) return;

	if (it->second->out_whc)
		dump_tensor_whc(filename, &it->second->output);
	else 
		dump_tensor_cwh(filename, &it->second->output);
}

void S16Network::dumpLayer(int idx0, int idx1, const char* filename) const
{
	auto it0 = d->layerMap.find(idx0);
	auto it1 = d->layerMap.find(idx1);
	if (it0 == d->layerMap.end() || it1 == d->layerMap.end()) return;
	if (it0->second->output.w != it1->second->output.w) return;
	if (it0->second->output.h != it1->second->output.h) return;
	if (it0->second->out_whc != it1->second->out_whc) return;

	int w = it0->second->output.w;
	int h = it0->second->output.h;
	int c0 = it0->second->output.c;
	int c1 = it1->second->output.c;
	S16Tensor merged = {
		w, h, (c0 + c1),
		w * h * (c0 + c1),
	};
	merged.ptr = (short*)malloc(sizeof(short) * merged.whc);
	assert(merged.ptr);
	memcpy(merged.ptr, it0->second->output.ptr, sizeof(short) * w * h * c0);
	memcpy(merged.ptr + w*h*c0, it1->second->output.ptr, sizeof(short) * w * h * c1);

	if (it0->second->out_whc)
		dump_tensor_whc(filename, &merged);
	else
		dump_tensor_cwh(filename, &merged);

	free(merged.ptr);
}
