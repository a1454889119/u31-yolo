#include <filesystem>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <UpixNetwork/Layers/UpixBaseLayer.h>
#include <UpixNetwork/UpixNetwork.h>
#include <UpixNetwork/TargetDecoder.h>

#define CV_VERSION_ID       CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)

#ifdef _DEBUG
#define cvLIB(name) "opencv_" name CV_VERSION_ID "d"
#else
#define cvLIB(name) "opencv_" name CV_VERSION_ID
#endif
#pragma comment(lib, cvLIB("world"))
#pragma comment(lib, "UpixNetwork.lib")

#define MODEL_PATH "../../model/"
#define DATA_PATH "../../data/"

static const cv::Scalar COLORS[] = {
	CV_RGB(0x00, 0xff, 0x00),
	CV_RGB(0xff, 0x00, 0x00),
	CV_RGB(0x00, 0x80, 0x00),
	CV_RGB(0x80, 0x00, 0x00),
	CV_RGB(0x00, 0xc0, 0x40),
	CV_RGB(0xc0, 0x40, 0x00),
};

//#define  C2_0_IDX 68
//#define C40_0_IDX 73
//#define  C2_1_IDX 93
//#define C40_1_IDX 98
//#define  C2_2_IDX 120
//#define C40_2_IDX 125
#define  C2_0_IDX 63
#define C40_0_IDX 68
#define  C2_1_IDX 89
#define C40_1_IDX 94
#define  C2_2_IDX 117
#define C40_2_IDX 122

static void run_image(UpixNetwork* net, cv::Mat& img)
{
	cv::Mat inputImg = img;
	if (img.cols == 320 && img.rows == 240) {
		cv::Mat up = img(cv::Rect(0, 0, 320, 40));
		cv::Mat down = img(cv::Rect(0, 200, 320, 40));
		up = 0;
		down = 0;
		inputImg = img(cv::Rect(0, 40, 320, 160));
	}

	cv::Mat imgfloat;
	if (inputImg.channels() == 1) {
		inputImg.convertTo(imgfloat, CV_32FC1);
	}
	else {
		cv::cvtColor(inputImg, imgfloat, cv::COLOR_BGR2GRAY);
		imgfloat.convertTo(imgfloat, CV_32FC1);
	}

	imgfloat *= 1.0f / 255.0f;
	net->forward((float*)imgfloat.data);
}

static void run_image_int(UpixNetwork* net, cv::Mat& img)
{
	cv::Mat inputImg = img;
	if (img.cols == 320 && img.rows == 240) {
		cv::Mat up = img(cv::Rect(0, 0, 320, 40));
		cv::Mat down = img(cv::Rect(0, 200, 320, 40));
		up = 0;
		down = 0;
		inputImg = img(cv::Rect(0, 40, 320, 160));
	}

	cv::Mat imgInt;
	if (inputImg.channels() == 1) {
		inputImg.convertTo(imgInt, CV_32SC1);
	}
	else {
		cv::cvtColor(inputImg, imgInt, cv::COLOR_BGR2GRAY);
		imgInt.convertTo(imgInt, CV_32SC1);
	}

	net->forward((int*)imgInt.data);
}


static void draw_rects(UpixNetwork* net, cv::Mat& img, int color_idx)
{
	TargetDecoder dec;
	dec.reset(0.25f);
	dec.decode(net->output(C2_0_IDX).fptr, net->output(C40_0_IDX).fptr, net->output(C2_0_IDX).w, net->output(C2_0_IDX).h, 4, 16);
	dec.decode(net->output(C2_1_IDX).fptr, net->output(C40_1_IDX).fptr, net->output(C2_1_IDX).w, net->output(C2_1_IDX).h, 4, 8);
	dec.decode(net->output(C2_2_IDX).fptr, net->output(C40_2_IDX).fptr, net->output(C2_2_IDX).w, net->output(C2_2_IDX).h, 4, 4);
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

static void draw_rects_int(UpixNetwork* net, cv::Mat& img, int color_idx)
{
	TargetDecoder dec;
	dec.reset(0.25f);
	dec.decode(net->output(C2_0_IDX).iptr, net->output(C40_0_IDX).iptr, net->output(C2_0_IDX).w, net->output(C2_0_IDX).h, 4, 16);
	dec.decode(net->output(C2_1_IDX).iptr, net->output(C40_1_IDX).iptr, net->output(C2_1_IDX).w, net->output(C2_1_IDX).h, 4, 8);
	dec.decode(net->output(C2_2_IDX).iptr, net->output(C40_2_IDX).iptr, net->output(C2_2_IDX).w, net->output(C2_2_IDX).h, 4, 4);
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

static std::vector<float> load_float(const char* filename)
{
	std::vector<float> data;
	FILE* fp = nullptr;
	fopen_s(&fp, filename, "rt");
	if (fp) {
		float fv;
		while (true) {
			if (1 != fscanf_s(fp, "%g\n", &fv))
				break;
			data.push_back(fv);
		}
		fclose(fp);
	}
	return data;
}


struct stDiff {
	float maxv;
	float avg;
	int idx;
};

static stDiff compare_data(const std::vector<float>& fdata, const float* fptr)
{
	stDiff diff = { 0.0f, 0.0f, -1 };
	float maxdiff = 0.0f;
	float sumdiff = 0.0f;
	int max_idx = -1;
	for (int i = 0; i < (int)fdata.size(); i++) {
		float fdiff = fabsf(fdata[i] - fptr[i]);
		sumdiff += fdiff;
		if (fdiff > diff.maxv) {
			diff.maxv = fdiff;
			diff.idx = i;
		}
	}
	diff.avg = sumdiff / fdata.size();
	return diff;
}

static void check_layer_output(const UpixBaseLayer* layer, const char* filename)
{
	std::vector<float> fdata = load_float(filename);
	if (fdata.empty()) {
		printf("Output of layer %d not found!\n", layer->info()->idx);
		return;
	}
	if (fdata.size() != layer->output().whc) {
		printf("layer %3d size NOT same!!!!\n", layer->info()->idx);
		return;
	}
	stDiff diff = compare_data(fdata, layer->output().fptr);
	printf("layer %3d, avgdiff:%g maxdiff:%g, value:%g, my value:%g\n",
		layer->info()->idx, diff.avg, diff.maxv, fdata[diff.idx], layer->output().fptr[diff.idx]);
}

static std::vector<std::string> load_layer_names(const char* filename)
{
	std::vector<std::string> result;
	std::ifstream ifs(filename);

	while (!ifs.eof()) {
		std::string line;
		getline(ifs, line);
		result.push_back(line);
	}
	ifs.close();

	return result;
}

static void check_network_results(const UpixNetwork& net, const char* result_path, const char* layer_names_file)
{
	auto layernames = load_layer_names(layer_names_file);
	//assert(layernames.size() == net.layers().size());
	for (auto layer : net.layers()) {
		if (layer->info()->idx < (int)layernames.size()) {
			std::string name = std::string(result_path) + "/" + layernames[layer->info()->idx] + ".txt";

			check_layer_output(layer, name.c_str());
		}
	}
}

static void dump_bin(const char* input_png, int w, int h, const char* output_bin)
{
	cv::Mat img = cv::imread(input_png, 0);
	cv::resize(img, img, cv::Size(w, h));
	FILE* fp = nullptr;
	fopen_s(&fp, output_bin, "wb");
	if (fp) {
		fwrite(img.data, img.cols, img.rows, fp);
		fclose(fp);
	}
}


static void dump_uint8_whc(const char* filename, TensorShape shape)
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

static void dump_uint8_cwh(const char* filename, TensorShape shape)
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

static void dump_int16_whc(const char* filename, TensorShape shape)
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

static void dump_int16_cwh(const char* filename, TensorShape shape)
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

static void dump_int_result(UpixNetwork& netq, const char* outpath)
{
	std::filesystem::path p(outpath);
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

static void test_pool_320x160()
{
	cv::Mat img = cv::imread(DATA_PATH"20260410_090428.png");
	cv::resize(img, img, cv::Size(320, 160));
	//dump_bin(DATA_PATH"pool.png", 320, 160, "pool_320x160.bin");

	cv::Mat input;
	cv::cvtColor(img, input, cv::COLOR_BGR2GRAY);
	UpixNetwork netq;
	UpixNetwork net;
	if (net.loadNetwork(MODEL_PATH"Pool_320x160_4classes.network.csv")
		&& net.loadWeight(MODEL_PATH"Pool_320x160_4classes.weights.txt"))
	{
		if (!netq.loadNetwork(MODEL_PATH"Pool_320x160_4classes.network.csv")) assert(0);
		if (!netq.loadWeight(MODEL_PATH"Pool_320x160_4classes.weights.txt")) assert(0);
		if (!netq.loadQuant(MODEL_PATH"Pool_320x160_4classes.maxmin.txt")) assert(0);
		//if (!netq.loadQuant(MODEL_PATH"320x160_maxmin.txt")) assert(0);
		//netq.dumpQuantedWeights("320x160_quanted_weights.txt");
		//dump_maxmin(net, "D:\\git\\u31-yolo\\yolo-c\\data\\val_UPIX_face_human_images_0915_person\\");
		//runImages(TEST_IMAGES_PATH, net, netq);
				
		

		run_image(&net, input);		
		run_image_int(&netq, input);

		//dump_int_result(netq, "output_int");

		draw_rects(&net, img, 0);		
		draw_rects_int(&netq, img, 4);
	}
}

static std::vector<std::string> get_img_full_names(const char* imgpath)
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

#define TEST_SET "\\\\192.168.1.57\\u31-ai人脸人行\\01-泳池数据\\V4模型验证集数据\\val_2class"
static void dump_maxmin()
{
	UpixNetwork net;
	if (net.loadNetwork(MODEL_PATH"320x160.network.csv")
		&& net.loadWeight(MODEL_PATH"320x160.weights.txt"))
	{
		auto names = get_img_full_names(TEST_SET);

		std::vector<stMaxMin> mms(net.layers().size());
		int total = (int)names.size();
		int i = 0;
		for (auto& name : names) {
			printf("%5d/%5d\r", i++, total);
			cv::Mat img = cv::imread(name.c_str(), 0);
			cv::resize(img, img, cv::Size(320, 160));
			run_image(&net, img);

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
		}
	}
}

int main(int argc, char** argv)
{
	//dump_maxmin();
	test_pool_320x160();
	system("pause");
	return 0;
}
