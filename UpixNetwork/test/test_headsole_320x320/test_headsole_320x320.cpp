#include <opencv2/opencv.hpp>
#include <UpixNetwork/Layers/UpixBaseLayer.h>
#include <UpixNetwork/UpixNetwork.h>
#include <UpixNetwork/TargetDecoder.h>
#include <utils/utils.h>
#include <algorithm>
#include <assert.h>

#define CV_VERSION_ID       CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)

#ifdef _DEBUG
#define cvLIB(name) "opencv_" name CV_VERSION_ID "d"
#else
#define cvLIB(name) "opencv_" name CV_VERSION_ID
#endif
#pragma comment(lib, cvLIB("world"))
#pragma comment(lib, "UpixNetwork.lib")
#pragma comment(lib, "UpixNetworkUtils.lib")

#define MODEL_DIR "D:/head_sole/ultralytics/runs/detect/train3/weights/"
#define DEFAULT_IMAGE "D:/head_sole/data/dataset_2026_05_26/images/UPIX_Raw_2026-05-07_15-17-17_00001.png"

static const char* CLASS_NAMES[] = {
	"front",
	"heel",
};

static const cv::Scalar COLORS[] = {
	CV_RGB(0x00, 0xff, 0x00),
	CV_RGB(0xff, 0x00, 0x00),
	CV_RGB(0x00, 0xc0, 0x40),
	CV_RGB(0xc0, 0x40, 0x00),
};

static const int TYPE_COUNT = 2;
static const int INPUT_W = 320;
static const int INPUT_H = 320;

#define  C2_0_IDX 43
#define C40_0_IDX 48
#define  C2_1_IDX 59
#define C40_1_IDX 64

static void run_image(UpixNetwork* net, UpixNetwork* qnet, cv::Mat& img, std::vector<Target>& ftargets, std::vector<Target>& itargets)
{
	cv::Mat inputImg = img;
	if (inputImg.cols != INPUT_W || inputImg.rows != INPUT_H) {
		cv::resize(inputImg, inputImg, cv::Size(INPUT_W, INPUT_H));
	}

	cv::Mat imgfloat;
	cv::Mat imgInt;
	if (inputImg.channels() == 1) {
		inputImg.convertTo(imgfloat, CV_32FC1);
		inputImg.convertTo(imgInt, CV_32SC1);
	}
	else {
		cv::cvtColor(inputImg, imgfloat, cv::COLOR_BGR2GRAY);
		imgfloat.convertTo(imgfloat, CV_32FC1);
		cv::cvtColor(inputImg, imgInt, cv::COLOR_BGR2GRAY);
		imgInt.convertTo(imgInt, CV_32SC1);
	}

	imgfloat *= 1.0f / 255.0f;
	net->forward((float*)imgfloat.data);
	qnet->forward((int*)imgInt.data);

	TargetDecoder dec;
	dec.reset(0.25f);
	dec.decode(net->output(C2_0_IDX).fptr, net->output(C40_0_IDX).fptr, net->output(C2_0_IDX).w, net->output(C2_0_IDX).h, TYPE_COUNT, 8);
	dec.decode(net->output(C2_1_IDX).fptr, net->output(C40_1_IDX).fptr, net->output(C2_1_IDX).w, net->output(C2_1_IDX).h, TYPE_COUNT, 4);
	ftargets = dec.getTargets();

	dec.reset(0.25f);
	dec.decode(qnet->output(C2_0_IDX).iptr, qnet->output(C40_0_IDX).iptr, qnet->output(C2_0_IDX).w, qnet->output(C2_0_IDX).h, TYPE_COUNT, 8);
	dec.decode(qnet->output(C2_1_IDX).iptr, qnet->output(C40_1_IDX).iptr, qnet->output(C2_1_IDX).w, qnet->output(C2_1_IDX).h, TYPE_COUNT, 4);
	itargets = dec.getTargets();
}

static void draw_targets(const std::vector<Target>& targets, cv::Mat& img, int color_idx)
{
	for (auto& tar : targets) {
		if (tar.type < 0 || tar.type >= TYPE_COUNT) {
			continue;
		}

		cv::Rect r(
			(int)(tar.x - tar.w * 0.5f),
			(int)(tar.y - tar.h * 0.5f),
			(int)tar.w,
			(int)tar.h);
		cv::rectangle(img, r, COLORS[tar.type + color_idx], 2);
		cv::putText(img, CLASS_NAMES[tar.type], cv::Point(r.x, std::max(12, r.y - 4)),
			cv::FONT_HERSHEY_SIMPLEX, 0.45, COLORS[tar.type + color_idx], 1);
		printf("%s type%d score%g x%g y%g w%g h%g\n",
			CLASS_NAMES[tar.type], tar.type, tar.score, tar.x, tar.y, tar.w, tar.h);
	}
}

static void test_image(const char* image_name)
{
	const char* network_name = MODEL_DIR"best.network.csv";
	const char* weights_name = MODEL_DIR"best.weights.txt";
	const char* maxmin_name = MODEL_DIR"best.maxmin.txt";

	UpixNetwork net;
	if (!net.loadNetwork(network_name)) assert(0);
	if (!net.loadWeight(weights_name)) assert(0);

	UpixNetwork netq;
	if (!netq.loadNetwork(network_name)) assert(0);
	if (!netq.loadWeight(weights_name)) assert(0);
	if (!netq.loadQuant(maxmin_name)) assert(0);

	cv::Mat img = cv::imread(image_name, cv::IMREAD_GRAYSCALE);
	if (img.empty()) {
		printf("load image failed: %s\n", image_name);
		return;
	}
	cv::resize(img, img, cv::Size(INPUT_W, INPUT_H));

	std::vector<Target> ftargets, itargets;
	run_image(&net, &netq, img, ftargets, itargets);

	cv::Mat draw;
	cv::cvtColor(img, draw, cv::COLOR_GRAY2BGR);
	printf("-----float result\n");
	draw_targets(ftargets, draw, 0);
	printf("-----int result\n");
	draw_targets(itargets, draw, 2);

	cv::imshow("head_sole_320x320", draw);
	cv::waitKey(0);
}

int main(int argc, char** argv)
{
	const char* image_name = argc > 1 ? argv[1] : DEFAULT_IMAGE;
	test_image(image_name);
	return 0;
}
