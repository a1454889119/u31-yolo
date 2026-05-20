#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <filesystem>
#include <set>
#include <opencv2/opencv.hpp>

#include <S16Network/S16Network.h>
#include <S16Network/layers/S16BaseLayer.h>
#include <S16Network/decode_rects.h>
#include <S16Network/decode_rects_160x120.h>

#define CV_VERSION_ID       CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)

#ifdef _DEBUG
#define cvLIB(name) "opencv_" name CV_VERSION_ID "d"
#else
#define cvLIB(name) "opencv_" name CV_VERSION_ID
#endif
#pragma comment(lib, cvLIB("world"))
#pragma comment(lib, "S16Network")

#pragma warning(disable:4996)

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

static std::vector<Target2Type> detect_image(S16Network* network, const short* input_data)
{
	std::vector<Target2Type> result;
	network->forward((short*)input_data);
	int n = decode_rects_160x120(
		network->getLayerOutput(43),
		network->getLayerOutput(48),
		network->layers()[43]->output.w,
		network->layers()[43]->output.h,

		network->getLayerOutput(59),
		network->getLayerOutput(64),
		network->layers()[59]->output.w,
		network->layers()[59]->output.h
	);

	const Target2Type* prect = get_decoded_rects_160x120();
	int i;
	for (i = 0; i < n; i++, prect++) {
		result.push_back(*prect);
	}
	return result;
}

static std::vector<Target2Type> detect_image_quant(S16Network* network, const short* input_data)
{
	std::vector<Target2Type> result;
	network->forward_quant8bit((short*)input_data);
	int n = decode_rects_160x120(
		network->getLayerOutput(43),
		network->getLayerOutput(48),
		network->layers()[43]->output.w,
		network->layers()[43]->output.h,

		network->getLayerOutput(59),
		network->getLayerOutput(64),
		network->layers()[59]->output.w,
		network->layers()[59]->output.h
	);

	const Target2Type* prect = get_decoded_rects_160x120();
	int i;
	for (i = 0; i < n; i++, prect++) {
		result.push_back(*prect);
	}
	return result;
}

static const cv::Scalar COLORS[] = {
	CV_RGB(0x00, 0xff, 0x00),
	CV_RGB(0xff, 0x00, 0x00),
	CV_RGB(0x00, 0x80, 0x00),
	CV_RGB(0x80, 0x00, 0x00),
};

static void draw_rects(cv::Mat& img, const std::vector<Target2Type>& rects)
{
	printf("16bit---\n");
	for (auto& r : rects) {
		cv::Rect cvr;
		cvr.x = (r.x - r.w / 2) >> 8;
		cvr.y = (r.y - r.h / 2) >> 8;
		cvr.y -= 16;
		cvr.width = r.w >> 8;
		cvr.height = r.h >> 8;
		printf("type:%d, score:%.3f, {%3d,%3d,%3d,%3d}\n", r.type, r.score/1024.0f, cvr.x, cvr.y, cvr.width, cvr.height);

		cv::rectangle(img, cvr, COLORS[r.type]);
	}
}

static void draw_rects_quant(cv::Mat& img, const std::vector<Target2Type>& rects)
{
	printf("8bit---\n");
	for (auto& r : rects) {
		cv::Rect cvr;
		cvr.x = (r.x - r.w / 2) >> 8;
		cvr.y = (r.y - r.h / 2) >> 8;
		cvr.y -= 16;
		cvr.width = r.w >> 8;
		cvr.height = r.h >> 8;
		printf("type:%d, score:%.3f, {%3d,%3d,%3d,%3d}\n", r.type, r.score/1024.0, cvr.x, cvr.y, cvr.width, cvr.height);

		cv::rectangle(img, cvr, COLORS[r.type + 2]);
	}
}

struct stLayerMaxMin {
	short maxv = SHRT_MIN;
	short minv = SHRT_MAX;
};
#define IMAGE_PATH "\\\\192.168.1.173\\优象资料库\\研发部\\研发部\\01-测试数据wdz\\06-YOLO数据集\\Png\\"


static void print_scales()
{
	std::set<int> notQuantLayers = {
		36, 37, 43, 48, 52, 53, 59, 64
	};
	S16Network* net = new S16Network;
	if (net->loadNetwork("../data/160x128.network.csv", 1)
		&& net->loadWeight("../data/160x128.weights.txt"))
	{
		std::vector<stLayerMaxMin> maxmins;
		for (int i = 0; i < (int)net->layers().size(); i++) {
			maxmins.push_back({ SHRT_MIN, SHRT_MAX });
		}

		short* input_data = (short*)malloc(sizeof(short) * net->inputW() * net->inputH() * net->inputC());
		assert(input_data);
		auto names = get_img_full_names(IMAGE_PATH);
		cv::Mat srcimg, dstimg(net->inputH(), net->inputW(), CV_8UC1);
		cv::Mat dstroi = dstimg(cv::Rect(0, 4, 160, 120));
		dstimg = 0;
		for (auto& name : names)
		{
			printf("%s\r", name.c_str());
			cv::Mat img = cv::imread(name, 0);
			cv::resize(img, dstroi, cv::Size(160, 120));
			get_input_data(dstimg, input_data);

			net->forward((short*)input_data);
			for (auto layer : net->layers())
			{
				auto& mm = maxmins[layer->idx];
				for (int i = 0; i < layer->output.whc; i++) {
					if (layer->output.ptr[i] > mm.maxv)
						mm.maxv = layer->output.ptr[i];
					else if (layer->output.ptr[i] < mm.minv)
						mm.minv = layer->output.ptr[i];
				}
			}
		}
		printf("\n");
		auto layers = net->layers();
		for (size_t i = 0; i < maxmins.size(); i++) {
			auto& mm = maxmins[i];
			if (notQuantLayers.count(i) == 0) {
				layers[i]->output.quants[0].scale = (mm.maxv - mm.minv);// / 256;
				layers[i]->output.quants[0].zero_point = (mm.maxv + mm.minv) / 2;
			}
			else {
				layers[i]->output.quants[0].scale = 1;
				layers[i]->output.quants[0].zero_point = 0;
			}
			printf("layer %2d: scale:%6d, zero_point:%6d\n", i,
				layers[i]->output.quants[0].scale,
				layers[i]->output.quants[0].zero_point);
		}
		free(input_data);
	}
	delete net;
}

static void load_scales(S16Network* network, const char* filename)
{
	FILE* fp = fopen(filename, "rt");
	assert(fp);
	auto layers = network->layers();
	int idx, scale, zero_point;
	std::set<int> result_layers = {
		43, 48, 59, 64
	};

	while (3 == fscanf(fp, "layer %2d: scale:%6d, zero_point:%6d\n",
		&idx, &scale, &zero_point))
	{
		if (scale == 1 && zero_point == 0 || result_layers.count(idx) == 1) {
			layers[idx]->output.quants[0].scale = 1;
			layers[idx]->output.quants[0].zero_point = zero_point;
		}
		else {
			layers[idx]->output.quants[0].scale = scale >> 8;
			layers[idx]->output.quants[0].zero_point = zero_point;
		}		
		layers[idx]->output.quant_count = 1;
	}
	fclose(fp);
}

static void compare_diff(S16Network* net1, S16Network* netq)
{
	auto layers1 = net1->layers();
	auto layers2 = netq->layers();
	assert(layers1.size() == layers2.size());
	for (size_t i = 0; i < layers1.size(); i++) {
		auto layer1 = layers1[i];
		auto layer2 = layers2[i];

		

		int scale = layer2->output.quants[0].scale;
		int zero_point = layer2->output.quants[0].zero_point;

		assert(layer1->output.whc == layer2->output.whc);
		int maxdiff = 0, sumdiff = 0;
		for (int k = 0; k < layer1->output.whc; k++) {
			// de-quantize
			int out2 = layer2->output.ptr[k] * scale + zero_point;
			int diff = abs(layer1->output.ptr[k] - out2);
			sumdiff += diff;
			if (diff > maxdiff) {
				maxdiff = diff;
			}
		}		
		printf("layer %2d: maxdiff:%6d, avgdiff:%4d\n", i, maxdiff, sumdiff / layer1->output.whc);
	}
}

static void compare_results()
{
	S16Network* net = new S16Network;
	S16Network* net_q = new S16Network;

	cv::namedWindow("test");
	if (net->loadNetwork("../data/160x128.network.csv", 1)
		&& net->loadWeight("../data/160x128.weights.txt"))
	{
		net_q->loadNetwork("../data/160x128.network.csv", 1);
		net_q->loadWeight("../data/160x128.weights.txt");

		load_scales(net_q, "scale.txt");
		short* input_data = (short*)malloc(sizeof(short) * net->inputW() * net->inputH() * net->inputC());
		assert(input_data);
		auto names = get_img_full_names(IMAGE_PATH);
		cv::Mat srcimg, dstimg(net->inputH(), net->inputW(), CV_8UC1);
		cv::Mat dstroi = dstimg(cv::Rect(0, 4, 160, 120));
		dstimg = 0;

		for (auto& name : names)
		{			
			printf("%s\n", name.c_str());

			cv::Mat img = cv::imread(name, 1);
			cv::Mat gray;
			cv::cvtColor(img, gray, CV_BGR2GRAY);
			cv::resize(gray, dstroi, cv::Size(160, 120));
			get_input_data(dstimg, input_data);

			auto rects_quant = detect_image_quant(net_q, input_data);
			//net->dumpToDir("8bitres");

			auto rects = detect_image(net, input_data);
			//net->dumpToDir("16bitres");



			compare_diff(net, net_q);

			draw_rects(img, rects);
			draw_rects_quant(img, rects_quant);
			cv::imshow("testquant", img);
			int key = cv::waitKey(0);
			if (key == 27) break;
		}

		free(input_data);
	}
	cv::destroyAllWindows();

	delete net;
}

static void test_cam()
{
	S16Network* net = new S16Network;
	net->loadNetwork("../data/160x128.network.csv", 1);
	net->loadWeight("../data/160x128.weights.txt");
	short* input_data = (short*)malloc(sizeof(short) * net->inputW() * net->inputH() * net->inputC());
	assert(input_data);

	cv::VideoCapture vc;
	if (vc.open(0)) {

		cv::Mat frame, img160x128(128, 160, CV_8UC1), small;
		img160x128 = 0;
		while (vc.read(frame)) {
			assert(frame.cols == 640 && frame.rows == 480);

			cv::Mat img160x120 = img160x128(cv::Rect(0, 8, 160, 120));
			cv::resize(frame, small, cv::Size(160, 120), 0, 0, 0);
			cv::cvtColor(small, img160x120, CV_BGR2GRAY);

			get_input_data(img160x128, input_data);


			auto rects = detect_image(net, input_data);

			//compare_diff(net, netq);
			draw_rects(frame, rects);
			cv::imshow("detect result", frame);
			cv::imshow("160x120", img160x128);
			
			int key = cv::waitKey(1);
			if (key == 27) break;
		}
	}
	cv::destroyAllWindows();

	free(input_data);
	delete net;
}

int main(int argc, char** argv)
{
	test_cam();
	//print_scales();
	//compare_results();
	return 0;
}