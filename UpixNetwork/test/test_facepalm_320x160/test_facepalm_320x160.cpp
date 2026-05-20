#include <filesystem>
#include <opencv2/opencv.hpp>
#include <UpixNetwork/Layers/UpixBaseLayer.h>
#include <UpixNetwork/UpixNetwork.h>
#include <UpixNetwork/TargetDecoder.h>
#include <utils/utils.h>

#define CV_VERSION_ID       CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)

#ifdef _DEBUG
#define cvLIB(name) "opencv_" name CV_VERSION_ID "d"
#else
#define cvLIB(name) "opencv_" name CV_VERSION_ID
#endif
#pragma comment(lib, cvLIB("world"))
#pragma comment(lib, "UpixNetwork.lib")
#pragma comment(lib, "UpixNetworkUtils.lib")

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

#define  C2_0_IDX 63
#define C40_0_IDX 68
#define  C2_1_IDX 89
#define C40_1_IDX 94
#define  C2_2_IDX 117
#define C40_2_IDX 122

static void run_image(UpixNetwork* net, UpixNetwork* qnet, cv::Mat& img, std::vector<Target>& ftargets, std::vector<Target>& itargets)
{
	cv::Mat inputImg = img;
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
	dec.decode(net->output(C2_0_IDX).fptr, net->output(C40_0_IDX).fptr, net->output(C2_0_IDX).w, net->output(C2_0_IDX).h, 3, 16);
	dec.decode(net->output(C2_1_IDX).fptr, net->output(C40_1_IDX).fptr, net->output(C2_1_IDX).w, net->output(C2_1_IDX).h, 3, 8);
	dec.decode(net->output(C2_2_IDX).fptr, net->output(C40_2_IDX).fptr, net->output(C2_2_IDX).w, net->output(C2_2_IDX).h, 3, 4);
	ftargets = dec.getTargets();
	dec.reset(0.25f);
	dec.decode(qnet->output(C2_0_IDX).iptr, qnet->output(C40_0_IDX).iptr, qnet->output(C2_0_IDX).w, qnet->output(C2_0_IDX).h, 3, 16);
	dec.decode(qnet->output(C2_1_IDX).iptr, qnet->output(C40_1_IDX).iptr, qnet->output(C2_1_IDX).w, qnet->output(C2_1_IDX).h, 3, 8);
	dec.decode(qnet->output(C2_2_IDX).iptr, qnet->output(C40_2_IDX).iptr, qnet->output(C2_2_IDX).w, qnet->output(C2_2_IDX).h, 3, 4);
	itargets = dec.getTargets();
}

static void draw_targets(const std::vector<Target>& targets, cv::Mat& img, int scale, int color_idx)
{
	for (auto& tar : targets) {
		cv::Rect r(
			(int)(tar.x - tar.w * 0.5f) * scale,
			(int)(tar.y - tar.h * 0.5f) * scale,
			(int)tar.w * scale,
			(int)tar.h * scale);
		cv::rectangle(img, r, COLORS[tar.type + color_idx]);
	}
}

static void test_cam640x480()
{
	UpixNetwork net;
	if (net.loadNetwork(MODEL_PATH"facepalm_320x160_3types.network.csv")
		&& net.loadWeight(MODEL_PATH"facepalm_320x160_3types.weights.txt"))
	{
		UpixNetwork netq;
		if (!netq.loadNetwork(MODEL_PATH"facepalm_320x160_3types.network.csv")) assert(0);
		if (!netq.loadWeight(MODEL_PATH"facepalm_320x160_3types.weights.txt")) assert(0);
		if (!netq.loadQuant(MODEL_PATH"facepalm_320x160_3types.maxmin.txt")) assert(0);
		cv::VideoCapture vc;
		if (vc.open(0)) {
			cv::Mat frame, gray;
			cv::Rect roi = cv::Rect(0, 80, 640, 320);
			while (vc.read(frame))
			{
				cv::Mat img0 = frame(roi);
				cv::cvtColor(img0, gray, cv::COLOR_BGR2GRAY);
				cv::resize(gray, gray, cv::Size(320, 160));

				std::vector<Target> ftargets, itargets;
				run_image(&net, &netq, gray, ftargets, itargets);
				cv::Mat img_draw = frame(roi);

				cv::rectangle(frame, roi, CV_RGB(0, 0, 0), 2);
				draw_targets(ftargets, img_draw, 2, 0);
				draw_targets(itargets, img_draw, 2, 2);

				cv::imshow("cam", frame);
				int key = cv::waitKey(1);
				if (key == 27) break;
			}
		}
	}
	cv::destroyAllWindows();
	system("pause");
}

#define VAL_IMAGE_PATH "\\\\192.168.1.57\\u31-ai人脸人行\\03-开发资料\\04-训练验证相关数据\\new_datasets\\facepalm\\images\\val"

int main(int argc, char** argv)
{
	//dump_maxmin(
	//	MODEL_PATH"facepalm_320x160.network.csv",
	//	MODEL_PATH"facepalm_320x160.weights.txt",
	//	VAL_IMAGE_PATH,
	//	MODEL_PATH"facepalm_320x160.maxmin.txt"
	//);

	test_cam640x480();
	return 0;
}