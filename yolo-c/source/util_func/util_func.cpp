#include <filesystem>
#include <S16Network/S16Network.h>
#include <S16Network/layers/S16BaseLayer.h>
#include "util_func.hpp"

#pragma warning(disable:4996)

std::vector<std::string> get_img_full_names(const char* imgpath)
{
	std::vector<std::string> names;
	auto dir = std::filesystem::directory_iterator(imgpath);
	for (auto& fe : dir) {
		auto fp = fe.path();
		if (fe.is_regular_file() && (fp.extension() == ".png" || fp.extension() == ".jpg")) {
			names.push_back(fp.string());
		}
	}
	return names;
}

void get_input_data(const cv::Mat& img, short* input_data)
{
	int wh = img.rows * img.cols;
	const unsigned char* src_ptr = img.data;

	if (img.channels() == 3) {
		short* input_b = input_data;
		short* input_g = input_data + wh;
		short* input_r = input_data + wh * 2;
		for (int i = 0; i < wh; i++) {
			input_b[i] = (short)src_ptr[0];
			input_g[i] = (short)src_ptr[1];
			input_r[i] = (short)src_ptr[2];
			src_ptr += 3;
		}
	}
	else {
		for (int i = 0; i < wh; i++) {
			input_data[i] = (short)src_ptr[i];
		}
	}
}

void image2bin(const cv::Mat& img, const char* filename)
{
	FILE* fp = fopen(filename, "wb");
	if (fp) {
		fwrite(img.data, img.cols, img.rows * img.channels(), fp);
		fclose(fp);
	}
}

std::vector<Quantize> load_scales(const char* filename)
{
	std::vector<Quantize> res;
	FILE* fp = fopen(filename, "rt");
	assert(fp);
	int idx, scale, zero_point;
	while (3 == fscanf(fp, "layer %3d: scale:%6d, zero_point:%6d\n",
		&idx, &scale, &zero_point))
	{
		assert(idx == (int)res.size());
		if (scale > 1)
			res.push_back({ scale >> 8, zero_point, 0, 0 });
		else 
			res.push_back({ 1, 0, 0, 0 });
	}
	fclose(fp);
	return res;
}


void set_scales(S16Network* network, const std::vector<Quantize>& scales)
{
	assert(scales.size() == network->layers().size());
	for (size_t i = 0; i < scales.size(); i++) {
		auto layer = network->layers()[i];
		auto& q = scales[i];
		layer->output.quants[0] = q;		
		layer->output.quant_count = 1;
	}

	for (auto layer : network->layers()) {
		layer->input.quant_count = layer->from_count;
		int c = 0;
		for (int i = 0; i < layer->from_count; i++) {
			const S16Tensor* t = layer->froms[i].t;
			layer->input.quants[i].scale = t->quants[0].scale;
			layer->input.quants[i].zero_point = t->quants[0].zero_point;
			layer->input.quants[i].c_start = c;
			layer->input.quants[i].c_end = c + layer->froms[i].channels;
		}
	}
}