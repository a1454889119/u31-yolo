#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <filesystem>
#include <set>
#include <opencv2/opencv.hpp>

#include <S16Network/S16Network.h>
#include <S16Network/layers/S16BaseLayer.h>
#include <S16Network/layers/S16ConvLayer.h>
#include <S16Network/OnnxDecoder.h>
#include <util_func/util_func.hpp>

#define CV_VERSION_ID       CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)

#ifdef _DEBUG
#define cvLIB(name) "opencv_" name CV_VERSION_ID "d"
#else
#define cvLIB(name) "opencv_" name CV_VERSION_ID
#endif
#pragma comment(lib, cvLIB("world"))
#pragma comment(lib, "S16Network")

#pragma warning(disable:4996)

#define NETWORK_CSV "../data/320x160.network.csv"
#define NETWORK_WEIGHTS "../data/320x160.weights.txt"

static std::vector<OnnxTarget> detect_image(S16Network* net, const short* input_data)
{
	std::vector<OnnxTarget> result;
	OnnxDecoder decoder(0.25f);
	net->forward((short*)input_data);
	decoder.decode(
		net->getLayerOutput(63),
		net->getLayerOutput(68),
		net->layers()[68]->output.w,
		net->layers()[68]->output.h,
		16
	);
	decoder.decode(
		net->getLayerOutput(89),
		net->getLayerOutput(94),
		net->layers()[94]->output.w,
		net->layers()[94]->output.h,
		8
	);
	decoder.decode(
		net->getLayerOutput(115),
		net->getLayerOutput(120),
		net->layers()[120]->output.w,
		net->layers()[120]->output.h,
		4
	);
	return decoder.getTargets();
}

static std::vector<OnnxTarget> detect_image_quant(S16Network* net, const short* input_data)
{
	std::vector<OnnxTarget> result;
	OnnxDecoder decoder(0.25f);
	net->forward_quant8bit((short*)input_data);
	decoder.decode(
		net->getLayerOutput(63),
		net->getLayerOutput(68),
		net->layers()[68]->output.w,
		net->layers()[68]->output.h,
		16
	);
	decoder.decode(
		net->getLayerOutput(89),
		net->getLayerOutput(94),
		net->layers()[94]->output.w,
		net->layers()[94]->output.h,
		8
	);
	decoder.decode(
		net->getLayerOutput(115),
		net->getLayerOutput(120),
		net->layers()[120]->output.w,
		net->layers()[120]->output.h,
		4
	);
	return decoder.getTargets();
}

static const cv::Scalar COLORS[] = {
	CV_RGB(0x00, 0xff, 0x00),
	CV_RGB(0xff, 0x00, 0x00),
	CV_RGB(0x00, 0x80, 0x00),
	CV_RGB(0x80, 0x00, 0x00),
};

static void draw_rects(cv::Mat& img, const std::vector<OnnxTarget>& rects)
{
	printf("16bit---\n");
	for (auto& r : rects) {
		cv::Rect cvr;
		cvr.x = (r.x - r.w / 2) >> 9;
		cvr.y = (r.y - r.h / 2) >> 9;
		cvr.y += 80;
		cvr.width = r.w >> 9;
		cvr.height = r.h >> 9;
		printf("type:%d, score:%.3f, {%3d,%3d,%3d,%3d}\n", r.type, r.score / 1024.0f, cvr.x, cvr.y, cvr.width, cvr.height);

		cv::rectangle(img, cvr, COLORS[r.type]);
	}
}

static void draw_rects_quant(cv::Mat& img, const std::vector<OnnxTarget>& rects)
{
	printf("8bit---\n");
	for (auto& r : rects) {
		cv::Rect cvr;
		cvr.x = (r.x - r.w / 2) >> 9;
		cvr.y = (r.y - r.h / 2) >> 9;
		cvr.y += 80;
		cvr.width = r.w >> 9;
		cvr.height = r.h >> 9;
		printf("type:%d, score:%.3f, {%3d,%3d,%3d,%3d}\n", r.type, r.score / 1024.0, cvr.x, cvr.y, cvr.width, cvr.height);

		cv::rectangle(img, cvr, COLORS[r.type + 2]);
	}
}

struct stLayerMaxMin {
	short maxv = SHRT_MIN;
	short minv = SHRT_MAX;
};
#define IMAGE_PATH "\\\\192.168.1.173\\优象资料库\\研发部\\研发部\\01-测试数据wdz\\06-YOLO数据集\\Png\\"

static bool check_layer_not_quant(const S16BaseLayer* layer)
{
	bool notQuant = false;
	switch (layer->type) {
	case S16Layer_Maxpool:
	case S16Layer_GlobalAvg:
	case S16Layer_Upsample:
		notQuant = true;
		break;
	case S16Layer_Convolution:
		if (layer->output.c == 2 || layer->output.c == 40) {
			notQuant = true;
		}
		else {
			S16ConvLayer* conv = (S16ConvLayer*)layer->priv;
			if (conv->activate_type == ActivateType_Sigmoid) {
				notQuant = true;
			}
		}
		break;

	default: break;
	}
	return notQuant;
}

static void print_scales()
{
	S16Network* net = new S16Network;
	if (net->loadNetwork(NETWORK_CSV, 1)
		&& net->loadWeight(NETWORK_WEIGHTS))
	{
		std::vector<stLayerMaxMin> maxmins;
		for (int i = 0; i < (int)net->layers().size(); i++) {
			maxmins.push_back({ SHRT_MIN, SHRT_MAX });
		}

		short* input_data = (short*)malloc(sizeof(short) * net->inputW() * net->inputH() * net->inputC());
		assert(input_data);
		auto names = get_img_full_names(IMAGE_PATH);
		cv::Mat srcimg, dstimg(net->inputH(), net->inputW(), CV_8UC1);
		for (auto& name : names)
		{
			printf("%s\r", name.c_str());
			cv::Mat img = cv::imread(name, 0);
			cv::Mat srcroi = img(cv::Rect(0, 80, 640, 320));
			cv::resize(srcroi, dstimg, cv::Size(320, 160));
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
			bool notQuant = check_layer_not_quant(layers[i]);
			if (notQuant) {
				printf("layer %3d: scale:%6d, zero_point:%6d\n", i, 1, 0);
			}
			else {
				auto& mm = maxmins[i];
				int scale = mm.maxv - mm.minv;
				int zero_point = mm.minv;
				printf("layer %3d: scale:%6d, zero_point:%6d\n", i, scale, zero_point);
			}
			
		}
		free(input_data);
	}
	delete net;
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
		printf("layer %3d: maxdiff:%6d(%.2f), avgdiff:%4d(%.2f)\n", i, maxdiff, maxdiff / 1024.0f, sumdiff / layer1->output.whc, sumdiff / layer1->output.whc / 1024.0f);
	}
}

static void compare_results()
{
	S16Network* net = new S16Network;
	S16Network* net_q = new S16Network;

	cv::namedWindow("testquant320x160");
	if (net->loadNetwork(NETWORK_CSV, 1)
		&& net->loadWeight(NETWORK_WEIGHTS))
	{
		net_q->loadNetwork(NETWORK_CSV, 1);
		net_q->loadWeight(NETWORK_WEIGHTS);

		set_scales(net_q, load_scales("../data/320x160_quant_scale.txt"));
		short* input_data = (short*)malloc(sizeof(short) * net->inputW() * net->inputH() * net->inputC());
		assert(input_data);
		auto names = get_img_full_names(IMAGE_PATH);
		cv::Mat srcimg, dstimg(net->inputH(), net->inputW(), CV_8UC1);

		for (auto& name : names)
		{
			printf("%s\n", name.c_str());

			cv::Mat img = cv::imread(name, 1);
			cv::Mat gray;
			cv::cvtColor(img, gray, CV_BGR2GRAY);

			cv::Mat srcroi = gray(cv::Rect(0, 80, 640, 320));
			cv::resize(srcroi, dstimg, cv::Size(320, 160));

			get_input_data(dstimg, input_data);

			auto rects = detect_image(net, input_data);
			//net->dumpToDir("16bitres");
			auto rects_quant = detect_image_quant(net_q, input_data);

			compare_diff(net, net_q);
			draw_rects(img, rects);
			draw_rects_quant(img, rects_quant);

			cv::imshow("testquant320x160", img);
			int key = cv::waitKey(0);
			if (key == 27) break;
		}

		free(input_data);
	}
	cv::destroyAllWindows();

	delete net;
}

static const std::map<int, std::string> resultNames = {
	{0, "_model.0_act_LeakyRelu_output_0"},
	{1, "_model.1_cv1_act_LeakyRelu_output_0"},
	{2, "_model.1_cv2_conv_Conv_output_0"},
	{3, "_model.2_cv1_act_LeakyRelu_output_0"},
	{4, "_model.2_m.0_pw1_act_LeakyRelu_output_0"},
	{5, "_model.2_m.0_dw1_act_LeakyRelu_output_0"},
	{6, "_model.2_m.0_pw2_act_LeakyRelu_output_0"},
	{7, "_model.2_m.0_dw2_act_LeakyRelu_output_0"},
	{8, "_model.2_m.1_pw1_act_LeakyRelu_output_0"},
	{9, "_model.2_m.1_dw1_act_LeakyRelu_output_0"},
	{10, "_model.2_m.1_pw2_act_LeakyRelu_output_0"},
	{11, "_model.2_m.1_dw2_act_LeakyRelu_output_0"},
	{12, "_model.2_cv2_act_LeakyRelu_output_0"},
	{13, "_model.3_act_LeakyRelu_output_0"},
	{14, "_model.4_act_LeakyRelu_output_0"},
	{15, "_model.5_cv1_act_LeakyRelu_output_0"},
	{16, "_model.5_m.0_pw1_act_LeakyRelu_output_0"},
	{17, "_model.5_m.0_dw1_act_LeakyRelu_output_0"},
	{18, "_model.5_m.0_pw2_act_LeakyRelu_output_0"},
	{19, "_model.5_m.0_dw2_act_LeakyRelu_output_0"},
	{20, "_model.5_m.1_pw1_act_LeakyRelu_output_0"},
	{21, "_model.5_m.1_dw1_act_LeakyRelu_output_0"},
	{22, "_model.5_m.1_pw2_act_LeakyRelu_output_0"},
	{23, "_model.5_m.1_dw2_act_LeakyRelu_output_0"},
	{24, "_model.5_m.2_pw1_act_LeakyRelu_output_0"},
	{25, "_model.5_m.2_dw1_act_LeakyRelu_output_0"},
	{26, "_model.5_m.2_pw2_act_LeakyRelu_output_0"},
	{27, "_model.5_m.2_dw2_act_LeakyRelu_output_0"},
	{28, "_model.5_m.3_pw1_act_LeakyRelu_output_0"},
	{29, "_model.5_m.3_dw1_act_LeakyRelu_output_0"},
	{30, "_model.5_m.3_pw2_act_LeakyRelu_output_0"},
	{31, "_model.5_m.3_dw2_act_LeakyRelu_output_0"},
	{32, "_model.5_cv2_act_LeakyRelu_output_0"},
	{33, "_model.6_cv1_act_LeakyRelu_output_0"},
	{34, "_model.6_m_MaxPool_output_0"},
	{35, "_model.6_m_1_MaxPool_output_0"},
	{36, "_model.6_m_2_MaxPool_output_0"},
	{37, "_model.6_cv2_act_LeakyRelu_output_0"},
	{38, "_model.7_act_LeakyRelu_output_0"},
	{39, "_model.8_act_LeakyRelu_output_0"},
	{40, "_model.9_cv1_act_LeakyRelu_output_0"},
	{41, "_model.9_m.0_pw1_act_LeakyRelu_output_0"},
	{42, "_model.9_m.0_dw1_act_LeakyRelu_output_0"},
	{43, "_model.9_m.0_pw2_act_LeakyRelu_output_0"},
	{44, "_model.9_m.0_dw2_act_LeakyRelu_output_0"},
	{45, "_model.9_m.1_pw1_act_LeakyRelu_output_0"},
	{46, "_model.9_m.1_dw1_act_LeakyRelu_output_0"},
	{47, "_model.9_m.1_pw2_act_LeakyRelu_output_0"},
	{48, "_model.9_m.1_dw2_act_LeakyRelu_output_0"},
	{49, "_model.9_m.2_pw1_act_LeakyRelu_output_0"},
	{50, "_model.9_m.2_dw1_act_LeakyRelu_output_0"},
	{51, "_model.9_m.2_pw2_act_LeakyRelu_output_0"},
	{52, "_model.9_m.2_dw2_act_LeakyRelu_output_0"},
	{53, "_model.9_cv2_act_LeakyRelu_output_0"},
	{54, "_model.10_act_LeakyRelu_output_0"},
	{55, "_model.11_act_LeakyRelu_output_0"},
	{56, "_model.12_pool_GlobalAveragePool_output_0"},
	{57, "_model.12_act_Sigmoid_output_0"},
	{58, "_model.12_Mul_output_0"},
	{59, "_model.25_cv3.0_cv3.0.0_act_LeakyRelu_output_0"},
	{60, "_model.25_cv3.0_cv3.0.1_conv_Conv_output_0"},
	{61, "_model.25_cv3.0_cv3.0.2_act_LeakyRelu_output_0"},
	{62, "_model.25_cv3.0_cv3.0.3_conv_Conv_output_0"},
	{63, "_model.25_cv3.0_cv3.0.4_Conv_output_0"},
	{64, "_model.25_cv2.0_cv2.0.0_act_LeakyRelu_output_0"},
	{65, "_model.25_cv2.0_cv2.0.1_conv_Conv_output_0"},
	{66, "_model.25_cv2.0_cv2.0.2_act_LeakyRelu_output_0"},
	{67, "_model.25_cv2.0_cv2.0.3_conv_Conv_output_0"},
	{68, "_model.25_cv2.0_cv2.0.4_Conv_output_0"},
	{69, "_model.13_Resize_output_0"},
	{70, "_model.15_act_LeakyRelu_output_0"},
	{71, "_model.16_act_LeakyRelu_output_0"},
	{72, "_model.17_cv1_act_LeakyRelu_output_0"},
	{73, "_model.17_m.0_pw1_act_LeakyRelu_output_0"},
	{74, "_model.17_m.0_dw1_act_LeakyRelu_output_0"},
	{75, "_model.17_m.0_pw2_act_LeakyRelu_output_0"},
	{76, "_model.17_m.0_dw2_act_LeakyRelu_output_0"},
	{77, "_model.17_m.1_pw1_act_LeakyRelu_output_0"},
	{78, "_model.17_m.1_dw1_act_LeakyRelu_output_0"},
	{79, "_model.17_m.1_pw2_act_LeakyRelu_output_0"},
	{80, "_model.17_m.1_dw2_act_LeakyRelu_output_0"},
	{81, "_model.17_cv2_act_LeakyRelu_output_0"},
	{82, "_model.18_pool_GlobalAveragePool_output_0"},
	{83, "_model.18_act_Sigmoid_output_0"},
	{84, "_model.18_Mul_output_0"},
	{85, "_model.25_cv3.1_cv3.1.0_act_LeakyRelu_output_0"},
	{86, "_model.25_cv3.1_cv3.1.1_conv_Conv_output_0"},
	{87, "_model.25_cv3.1_cv3.1.2_act_LeakyRelu_output_0"},
	{88, "_model.25_cv3.1_cv3.1.3_conv_Conv_output_0"},
	{89, "_model.25_cv3.1_cv3.1.4_Conv_output_0"},
	{90, "_model.25_cv2.1_cv2.1.0_act_LeakyRelu_output_0"},
	{91, "_model.25_cv2.1_cv2.1.1_conv_Conv_output_0"},
	{92, "_model.25_cv2.1_cv2.1.2_act_LeakyRelu_output_0"},
	{93, "_model.25_cv2.1_cv2.1.3_conv_Conv_output_0"},
	{94, "_model.25_cv2.1_cv2.1.4_Conv_output_0"},
	{95, "_model.19_Resize_output_0"},
	{96, "_model.21_act_LeakyRelu_output_0"},
	{97, "_model.22_act_LeakyRelu_output_0"},
	{98, "_model.23_cv1_act_LeakyRelu_output_0"},
	{99, "_model.23_m.0_pw1_act_LeakyRelu_output_0"},
	{100, "_model.23_m.0_dw1_act_LeakyRelu_output_0"},
	{101, "_model.23_m.0_pw2_act_LeakyRelu_output_0"},
	{102, "_model.23_m.0_dw2_act_LeakyRelu_output_0"},
	{103, "_model.23_m.1_pw1_act_LeakyRelu_output_0"},
	{104, "_model.23_m.1_dw1_act_LeakyRelu_output_0"},
	{105, "_model.23_m.1_pw2_act_LeakyRelu_output_0"},
	{106, "_model.23_m.1_dw2_act_LeakyRelu_output_0"},
	{107, "_model.23_cv2_act_LeakyRelu_output_0"},
	{108, "_model.24_pool_GlobalAveragePool_output_0"},
	{109, "_model.24_act_Sigmoid_output_0"},
	{110, "_model.24_Mul_output_0"},
	{111, "_model.25_cv3.2_cv3.2.0_act_LeakyRelu_output_0"},
	{112, "_model.25_cv3.2_cv3.2.1_conv_Conv_output_0"},
	{113, "_model.25_cv3.2_cv3.2.2_act_LeakyRelu_output_0"},
	{114, "_model.25_cv3.2_cv3.2.3_conv_Conv_output_0"},
	{115, "_model.25_cv3.2_cv3.2.4_Conv_output_0"},
	{116, "_model.25_cv2.2_cv2.2.0_act_LeakyRelu_output_0"},
	{117, "_model.25_cv2.2_cv2.2.1_conv_Conv_output_0"},
	{118, "_model.25_cv2.2_cv2.2.2_act_LeakyRelu_output_0"},
	{119, "_model.25_cv2.2_cv2.2.3_conv_Conv_output_0"},
	{120, "_model.25_cv2.2_cv2.2.4_Conv_output_0"},
};

struct stDiff {
	float maxv;
	float avg;
	int idx;
};

static stDiff compare_data(const std::vector<float>& fdata, const s16* sdata, int count)
{
	stDiff diff = { 0.0f, 0.0f, -1 };
	float maxdiff = 0.0f;
	float sumdiff = 0.0f;
	int max_idx = -1;
	for (int i = 0; i < count; i++) {
		float fdiff = fabsf(fdata[i] - ((float)sdata[i] / S16_QUANT_MULT));
		sumdiff += fdiff;
		if (fdiff > diff.maxv) {
			diff.maxv = fdiff;
			diff.idx = i;
		}
	}
	diff.avg = sumdiff / count;
	return diff;
}

static std::vector<float> load_float(const char* filename)
{
	std::vector<float> data;
	FILE* fp = fopen(filename, "rt");
	if (fp) {
		float fv;
		while (true) {
			if (1 != fscanf(fp, "%g\n", &fv))
				break;
			data.push_back(fv);
		}
		fclose(fp);
	}
	return data;
}

static void check_layer_output(const S16BaseLayer* layer, const char* filename)
{
	std::vector<float> fdata = load_float(filename);
	if (fdata.size() != layer->output.whc) {
		printf("layer %2d size NOT same!!!!\n", layer->idx);
		return;
	}
	stDiff diff = compare_data(fdata, layer->output.ptr, layer->output.whc);
	printf("layer %2d, avgdiff:%g maxdiff:%g, float value:%g, s16 value:%d\n",
		layer->idx, diff.avg, diff.maxv, fdata[diff.idx], layer->output.ptr[diff.idx]);
}

static void run_one_image()
{
	const std::string input_name = "3.png";
	const std::string result_path = "./1_result_float/";
	S16Network* net = new S16Network;
	net->loadNetwork("../data/320x160.network.csv", 1);
	net->loadWeight("../data/320x160.weights.txt");
	S16Network* netq = new S16Network;
	netq->loadNetwork("../data/320x160.network.csv", 1);
	netq->loadWeight("../data/320x160.weights.txt");
	set_scales(netq, load_scales("../data/320x160_quant_scale.txt"));

	cv::Mat img_bgr = cv::imread(input_name, 1);
	cv::Mat img_show(480, 640, CV_8UC3);
	cv::resize(img_bgr, img_show(cv::Rect(0, 80, 640, 320)), cv::Size(640, 320));
	cv::Mat img;
	cv::cvtColor(img_bgr, img, CV_BGR2GRAY);
	

	short* input_data = (short*)malloc(sizeof(short) * net->inputW() * net->inputH() * net->inputC());
	assert(input_data);

	get_input_data(img, input_data);
	net->forward(input_data);

	//image2bin(img, "320x160.bin");
	//net->dumpToDir("result_s8", 1);

	OnnxDecoder decoder(0.20f);
	decoder.reset();
	decoder.decode(
		net->getLayerOutput(63),
		net->getLayerOutput(68),
		net->layers()[68]->output.w,
		net->layers()[68]->output.h,
		16
	);
	decoder.decode(
		net->getLayerOutput(89),
		net->getLayerOutput(94),
		net->layers()[94]->output.w,
		net->layers()[94]->output.h,
		8
	);
	decoder.decode(
		net->getLayerOutput(115),
		net->getLayerOutput(120),
		net->layers()[120]->output.w,
		net->layers()[120]->output.h,
		4
	);
	draw_rects(img_show, decoder.getTargets());

	netq->forward_quant8bit(input_data);
	decoder.reset();
	decoder.decode(
		netq->getLayerOutput(63),
		netq->getLayerOutput(68),
		netq->layers()[68]->output.w,
		netq->layers()[68]->output.h,
		16
	);
	decoder.decode(
		netq->getLayerOutput(89),
		netq->getLayerOutput(94),
		netq->layers()[94]->output.w,
		netq->layers()[94]->output.h,
		8
	);
	decoder.decode(
		netq->getLayerOutput(115),
		netq->getLayerOutput(120),
		netq->layers()[120]->output.w,
		netq->layers()[120]->output.h,
		4
	);
	draw_rects_quant(img_show, decoder.getTargets());
	compare_diff(net, netq);

	//for (auto layer : net->layers()) {
	//	auto iter = resultNames.find(layer->idx);
	//	assert(iter != resultNames.end());
	//	auto filename = result_path + iter->second + ".txt";
	//	check_layer_output(layer, filename.c_str());
	//}
	free(input_data);
	delete net;
	delete netq;
}

static void run_images_cropped(const char* path)
{
	S16Network* net = new S16Network;
	net->loadNetwork("../data/320x160.network.csv", 1);
	net->loadWeight("../data/320x160.weights.txt");
	S16Network* netq = new S16Network;
	netq->loadNetwork("../data/320x160.network.csv", 1);
	netq->loadWeight("../data/320x160.weights.txt");
	set_scales(netq, load_scales("../data/320x160_quant_scale.txt"));
	short* input_data = (short*)malloc(sizeof(short) * net->inputW() * net->inputH() * net->inputC());
	assert(input_data);

	auto names = get_img_full_names(path);
	cv::Mat srcimg, dstimg(net->inputH(), net->inputW(), CV_8UC1);

	for (auto& name : names)
	{
		printf("%s\n", name.c_str());
		cv::Mat img_bgr = cv::imread(name, 1);
		cv::Mat img_show(480, 640, CV_8UC3);
		cv::resize(img_bgr, img_show(cv::Rect(0, 80, 640, 320)), cv::Size(640, 320));
		cv::Mat img;
		cv::cvtColor(img_bgr, img, CV_BGR2GRAY);

		get_input_data(img, input_data);
		
		
		auto rects = detect_image(net, input_data);
		//net->dumpToDir("16bitres");
		auto rects_quant = detect_image_quant(netq, input_data);

		//compare_diff(net, netq);
		draw_rects(img_show, rects);
		draw_rects_quant(img_show, rects_quant);

		cv::imshow("testquant320x160", img_show);
		int key = cv::waitKey(0);
		if (key == 27) break;

	}
	cv::destroyAllWindows();

	free(input_data);
	delete net;
	delete netq;
}

static void test_cam()
{
	S16Network* net = new S16Network;
	net->loadNetwork("../data/320x160.network.csv", 1);
	net->loadWeight("../data/320x160.weights.txt");
	S16Network* netq = new S16Network;
	netq->loadNetwork("../data/320x160.network.csv", 1);
	netq->loadWeight("../data/320x160.weights.txt");
	set_scales(netq, load_scales("../data/320x160_quant_scale.txt"));
	short* input_data = (short*)malloc(sizeof(short) * net->inputW() * net->inputH() * net->inputC());
	assert(input_data);

	cv::VideoCapture vc;
	if (vc.open(0)) {

		cv::Mat frame;
		while (vc.read(frame)) {
			assert(frame.cols == 640 && frame.rows == 480);
			cv::Mat top = frame(cv::Rect(0, 0, 640, 80));
			cv::Mat bot = frame(cv::Rect(0, 400, 640, 80));
			top = 0;
			bot = 0;

			cv::Mat srcroi = frame(cv::Rect(0, 80, 640, 320));
			cv::Mat dstimg;
			cv::resize(srcroi, dstimg, cv::Size(320, 160));
			cv::cvtColor(dstimg, dstimg, CV_BGR2GRAY);


			get_input_data(dstimg, input_data);


			auto rects = detect_image(net, input_data);
			//net->dumpToDir("16bitres");
			auto rects_quant = detect_image_quant(netq, input_data);

			//compare_diff(net, netq);
			draw_rects(frame, rects);
			draw_rects_quant(frame, rects_quant);
			cv::imshow("testquant320x160", frame);
			int key = cv::waitKey(1);
			if (key == 27) break;


		}
	}
	cv::destroyAllWindows();

	free(input_data);
	delete net;
	delete netq;
}

int main(int argc, char** argv)
{
	//test_cam();
	run_one_image();
	//run_images_cropped("06-2025-09-26有问题的测试场景/原图_cropped");
	//print_scales();
	//compare_results();
	system("pause");
	return 0;
}