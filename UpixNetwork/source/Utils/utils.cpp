#include <fstream>
#include <filesystem>
#include <Windows.h>
#include <UpixNetwork/UpixNetwork.h>
#include <UpixNetwork/Layers/UpixBaseLayer.h>
#include <UpixNetwork/UpixLayerInfo.h>
#include <UpixNetwork/TargetDecoder.h>

#include "utils.h"

std::vector<std::string> load_output_names(const char* filename)
{
	std::vector<std::string> results;
	std::ifstream ifs(filename);

	while (!ifs.eof()) {
		std::string line;
		getline(ifs, line);
		results.push_back(line);
	}
	ifs.close();
	return results;
}

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

void run_image_float(UpixNetwork* net, cv::Mat& img)
{
	cv::Mat inputImg = img;
	auto info = net->layers()[0]->info();
	if (info->inw != img.cols || info->inh != img.rows) {
		cv::resize(img, inputImg, cv::Size(info->inw, info->inh));
	}

	if (info->inc != img.channels()) {
		if (info->inc == 1) {
			cv::cvtColor(inputImg, inputImg, cv::COLOR_BGR2GRAY);
		}
		else if (info->inc == 3) {
			cv::cvtColor(inputImg, inputImg, cv::COLOR_GRAY2RGB);
		}
		else {
			printf("???Unknown channels!!!\n");
			exit(0);
		}
	}

	cv::Mat imgfloat;
	inputImg.convertTo(imgfloat, CV_32FC1);
	imgfloat *= 1.0f / 255.0f;
	net->forward((float*)imgfloat.data);
}

void run_image_int(UpixNetwork* net, cv::Mat& img)
{
	cv::Mat inputImg = img;
	auto info = net->layers()[0]->info();
	if (info->inw != img.cols || info->inh != img.rows) {
		cv::resize(img, inputImg, cv::Size(info->inw, info->inh));
	}

	if (info->inc != img.channels()) {
		if (info->inc == 1) {
			cv::cvtColor(inputImg, inputImg, cv::COLOR_BGR2GRAY);
		}
		else if (info->inc == 3) {
			cv::cvtColor(inputImg, inputImg, cv::COLOR_GRAY2RGB);
		}
		else {
			printf("???Unknown channels!!!\n");
			exit(0);
		}
	}

	cv::Mat imgInt;
	inputImg.convertTo(imgInt, CV_32SC1);
	net->forward((int*)imgInt.data);
}

static void get_max_min(const float* fptr, int count, float& maxf, float& minf)
{
	int i;
	for (i = 0; i < count; i++) {
		if (fptr[i] > maxf) maxf = fptr[i];
		else if (fptr[i] < minf) minf = fptr[i];
	}
}

struct stMaxMin {
	float maxv = -FLT_MAX;
	float minv = FLT_MAX;
};
bool dump_maxmin(
	const char* networkfilename,
	const char* weightsfilename,
	const char* imagepath,
	const char* outputfilename)
{
	bool res = false;
	UpixNetwork net;
	if (net.loadNetwork(networkfilename)
		&& net.loadWeight(weightsfilename))
	{
		FILE* fp = nullptr;
		fopen_s(&fp, outputfilename, "wt");
		if (fp) {
			auto names = get_img_full_names(imagepath);
			printf("%s目录下有%d个图片\n", imagepath, (int)names.size());

			std::vector<stMaxMin> mms(net.layers().size());
			int total = (int)names.size();
			int i = 0;
			for (auto& name : names) {
				printf("%5d/%5d\r", i++, total);
				cv::Mat img = cv::imread(name.c_str(), 0);
				cv::resize(img, img, cv::Size(320, 160));
				run_image_float(&net, img);

				for (auto layer : net.layers()) {
					auto shape = layer->output();
					auto& mm = mms[layer->info()->idx];
					get_max_min(shape.fptr, shape.whc, mm.maxv, mm.minv);
				}
			}
			printf("\n");

			for (int i = 0; i < (int)mms.size(); i++) {
				auto& mm = mms[i];
				printf("layer%d maxv:%g minv:%g\n", i, mm.maxv * 1.02f, mm.minv * 1.02f);
				fprintf(fp, "layer%d maxv:%g minv:%g\n", i, mm.maxv * 1.02f, mm.minv * 1.02f);
			}
			fclose(fp);
			res = true;
		}
		else {
			printf("打开文件%s失败!\n", outputfilename);
		}
	}
	else {
		printf("加载模型文件(%s,%s)失败!\n", networkfilename, weightsfilename);
	}
	return res;
}

static const cv::Scalar COLORS[] = {
	CV_RGB(0x00, 0xff, 0x00),
	CV_RGB(0xff, 0x00, 0x00),
	CV_RGB(0x00, 0x00, 0xff),
	CV_RGB(0x00, 0xc0, 0x00),
	CV_RGB(0xc0, 0x00, 0x00),
	CV_RGB(0x00, 0x00, 0xc0),
	CV_RGB(0x40, 0xd0, 0x00),
	CV_RGB(0xd0, 0x40, 0x00),
	CV_RGB(0x40, 0x00, 0xd0),
	CV_RGB(0x20, 0xe0, 0x20),
	CV_RGB(0xe0, 0x20, 0x20),
	CV_RGB(0x20, 0x20, 0xe0),
};

static float s_thres = 0.25f;
static int s_type_count = 0;
static std::vector<DecodeInfo> s_dec_infos;

void setDecodeInfo(int type_count, float thres, const std::vector<DecodeInfo>& dec_infos)
{
	s_type_count = type_count;
	s_dec_infos = dec_infos;
	s_thres = thres;
}

void draw_rects_float(UpixNetwork* net, cv::Mat& img, int color_idx)
{
	TargetDecoder dec;
	dec.reset(s_thres);

	for (auto& decInfo : s_dec_infos) {
		auto scoreShape = net->output(decInfo.score_layer);
		auto rectShape = net->output(decInfo.rect_layer);
		dec.decode(scoreShape.fptr, rectShape.fptr, scoreShape.w, scoreShape.h, s_type_count, decInfo.scale);
	}
	printf("-----float result\n");
	for (auto& tar : dec.getTargets()) {
		cv::Rect r(
			(int)(tar.x - tar.w * 0.5f),
			(int)(tar.y - tar.h * 0.5f),
			(int)tar.w,
			(int)tar.h);
		if (img.rows > 160) {
			r.y += (img.rows - 160) / 2;
		}
		cv::rectangle(img, r, COLORS[tar.type + color_idx]);
		printf("type%d, score%g\n", tar.type, tar.score);
	}
}


void draw_rects_int(UpixNetwork* net, cv::Mat& img, int color_idx)
{
	TargetDecoder dec;
	dec.reset(s_thres);
	for (auto& decInfo : s_dec_infos) {
		auto scoreShape = net->output(decInfo.score_layer);
		auto rectShape = net->output(decInfo.rect_layer);
		dec.decode(scoreShape.iptr, rectShape.iptr, scoreShape.w, scoreShape.h, s_type_count, decInfo.scale);
	}
	printf("-----int result\n");
	for (auto& tar : dec.getTargets()) {
		cv::Rect r(
			(int)(tar.x - tar.w * 0.5f),
			(int)(tar.y - tar.h * 0.5f),
			(int)tar.w,
			(int)tar.h);
		if (img.rows > 160) {
			r.y += (img.rows - 160) / 2;
		}
		cv::rectangle(img, r, COLORS[tar.type + color_idx]);
		printf("type%d, score%g\n", tar.type, tar.score);
	}
}

void dump_uint8_whc(const char* filename, TensorShape shape)
{
	FILE* fp = nullptr;
	fopen_s(&fp, filename, "wt");
	if (fp) {
		const int* ptr = shape.iptr;
		int i;
		for (i = 0; i < shape.whc; i++, ptr++) {
			fprintf(fp, "%02X ", (unsigned char)*ptr);
			if ((i + 1) % 32 == 0) fprintf(fp, "\n");
		}
		fclose(fp);
	}
}

void dump_uint8_cwh(const char* filename, TensorShape shape)
{
	FILE* fp = nullptr;
	fopen_s(&fp, filename, "wt");
	if (fp) {
		const int* ptr = shape.iptr;
		int i, j, k = 0;
		for (i = 0; i < shape.wh; i++, ptr++) {
			const int* cptr = ptr;
			for (j = 0; j < shape.c; j++, k++, cptr += shape.wh) {
				fprintf(fp, "%02X ", (unsigned char)*cptr);
				if ((k + 1) % 32 == 0) fprintf(fp, "\n");
			}
		}
		fclose(fp);
	}
}

void dump_int16_whc(const char* filename, TensorShape shape)
{
	FILE* fp = nullptr;
	fopen_s(&fp, filename, "wt");
	if (fp) {
		const int* ptr = shape.iptr;
		int i;
		for (i = 0; i < shape.whc; i++, ptr++) {
			fprintf(fp, "%04X ", (unsigned short)*ptr);
			if ((i + 1) % 16 == 0) fprintf(fp, "\n");
		}
		fclose(fp);
	}
}

void dump_int16_cwh(const char* filename, TensorShape shape)
{
	FILE* fp = nullptr;
	fopen_s(&fp, filename, "wt");
	if (fp) {
		const int* ptr = shape.iptr;
		int i, j, k = 0;
		for (i = 0; i < shape.wh; i++, ptr++) {
			const int* cptr = ptr;
			for (j = 0; j < shape.c; j++, k++, cptr += shape.wh) {
				fprintf(fp, "%04X ", (unsigned short)*cptr);
				if ((k + 1) % 16 == 0) fprintf(fp, "\n");
			}
		}
		fclose(fp);
	}
}

void dump_int16_result(UpixNetwork* netq, const char* outpath)
{
	std::filesystem::path p(outpath);
	if (!std::filesystem::exists(p)) {
		CreateDirectoryA(outpath, nullptr);
	}
	for (auto layer : netq->layers()) {
		auto info = layer->info();

		auto whc_name = (p / (std::to_string(info->idx) + "_out_whc.txt")).string();
		auto cwh_name = (p / (std::to_string(info->idx) + "_out_cwh.txt")).string();
		switch (info->outType) {
		case OutputType::WHC:
			dump_int16_whc(whc_name.c_str(), layer->output());
			break;
		case OutputType::CWH:
			dump_int16_cwh(cwh_name.c_str(), layer->output());
			break;

		case OutputType::WHC_CWH:
			dump_int16_whc(whc_name.c_str(), layer->output());
			dump_int16_cwh(cwh_name.c_str(), layer->output());
			break;
		default: break;
		}

		if (info->isrcs.size() > 1
			&& info->type == UpixLayerType::Convolution
			&& info->ksize == 1)
		{
			cwh_name = (p / (std::to_string(info->idx) + "_in16_cwh.txt")).string();
			dump_int16_cwh(cwh_name.c_str(), layer->input());
		}
	}
}

void dump_int8_result(UpixNetwork& netq, const char* outpath)
{
	std::filesystem::path p(outpath);
	if (!std::filesystem::exists(p)) {
		CreateDirectoryA(outpath, nullptr);
	}
	for (auto layer : netq.layers()) {
		auto info = layer->info();

		auto whc_name = (p / (std::to_string(info->idx) + "_out_whc.txt")).string();
		auto cwh_name = (p / (std::to_string(info->idx) + "_out_cwh.txt")).string();
		switch (info->outType) {
		case OutputType::WHC:
			if (info->quantInt.quantFlag == QuantIntFlag::MulSub) {
				dump_uint8_whc(whc_name.c_str(), layer->output());
			}
			else if (info->quantInt.quantFlag == QuantIntFlag::ScaleOnly) {
				dump_int16_whc(whc_name.c_str(), layer->output());
			}
			else {
				dump_uint8_whc(whc_name.c_str(), layer->output());
			}
			break;
		case OutputType::CWH:
			if (info->quantInt.quantFlag == QuantIntFlag::MulSub) {
				dump_uint8_cwh(cwh_name.c_str(), layer->output());
			}
			else {
				dump_int16_cwh(cwh_name.c_str(), layer->output());
			}
			break;

		case OutputType::WHC_CWH:
			if (info->quantInt.quantFlag == QuantIntFlag::MulSub) {
				dump_uint8_whc(whc_name.c_str(), layer->output());
				dump_uint8_cwh(cwh_name.c_str(), layer->output());
			}
			else {
				dump_int16_whc(whc_name.c_str(), layer->output());
				dump_int16_cwh(cwh_name.c_str(), layer->output());
			}
			break;
		default: break;
		}

		if (info->isrcs.size() > 1
			&& info->type == UpixLayerType::Convolution
			&& info->ksize == 1)
		{
			cwh_name = (p / (std::to_string(info->idx) + "_in16_cwh.txt")).string();
			dump_int16_cwh(cwh_name.c_str(), layer->input());
			cwh_name = (p / (std::to_string(info->idx) + "_in8_cwh.txt")).string();
			dump_uint8_cwh(cwh_name.c_str(), layer->input());
		}
	}
}