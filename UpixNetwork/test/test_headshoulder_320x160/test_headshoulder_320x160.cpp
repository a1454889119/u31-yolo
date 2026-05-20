#include <opencv2/opencv.hpp>
#include <UpixNetwork/Layers/UpixBaseLayer.h>
#include <UpixNetwork/UpixNetwork.h>
#include <UpixNetwork/TargetDecoder.h>
#include <utils/utils.h>
#include "KalmanTracker.h"
#include "KalmanBoxTracker.h"

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

#define  C2_0_IDX 43
#define C40_0_IDX 48
#define  C2_1_IDX 59
#define C40_1_IDX 64

static void run_image(UpixNetwork* net, UpixNetwork* qnet, cv::Mat& img, int types, std::vector<Target>& ftargets, std::vector<Target>& itargets)
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
	dec.decode(net->output(C2_0_IDX).fptr, net->output(C40_0_IDX).fptr, net->output(C2_0_IDX).w, net->output(C2_0_IDX).h, types, 8);
	dec.decode(net->output(C2_1_IDX).fptr, net->output(C40_1_IDX).fptr, net->output(C2_1_IDX).w, net->output(C2_1_IDX).h, types, 4);
	ftargets = dec.getTargets();
	dec.reset(0.25f);
	dec.decode(qnet->output(C2_0_IDX).iptr, qnet->output(C40_0_IDX).iptr, qnet->output(C2_0_IDX).w, qnet->output(C2_0_IDX).h, types, 8);
	dec.decode(qnet->output(C2_1_IDX).iptr, qnet->output(C40_1_IDX).iptr, qnet->output(C2_1_IDX).w, qnet->output(C2_1_IDX).h, types, 4);
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
	if (net.loadNetwork(MODEL_PATH"HeadShoulder_320x160.network.csv")
		&& net.loadWeight(MODEL_PATH"HeadShoulder_320x160.weights.txt"))
	{
		UpixNetwork netq;
		if (!netq.loadNetwork(MODEL_PATH"HeadShoulder_320x160.network.csv")) assert(0);
		if (!netq.loadWeight(MODEL_PATH"HeadShoulder_320x160.weights.txt")) assert(0);
		if (!netq.loadQuant(MODEL_PATH"HeadShoulder_320x160.maxmin.txt")) assert(0);
		cv::VideoCapture vc;
		if (vc.open(1)) {
			cv::Mat frame, gray;
			cv::Rect roi = cv::Rect(0, 80, 640, 320);
			while (vc.read(frame))
			{
				cv::Mat img0 = frame(roi);
				cv::cvtColor(img0, gray, cv::COLOR_BGR2GRAY);
				cv::resize(gray, gray, cv::Size(320, 160));

				std::vector<Target> ftargets, itargets;
				run_image(&net, &netq, gray, 4, ftargets, itargets);
				cv::Mat img_draw = frame(roi);

				cv::rectangle(frame, roi, CV_RGB(0, 0, 0), 2);
				draw_targets(ftargets, img_draw, 2, 0);
				draw_targets(itargets, img_draw, 2, 2);

				if (itargets.size() == 3) {
					int a = 0;
				}

				cv::imshow("cam", frame);
				int key = cv::waitKey(1);
				if (key == 27) break;
			}
		}
	}
	cv::destroyAllWindows();
	system("pause");
}

static void dump_image(cv::Mat& img, const char* filename)
{
	FILE* fp = nullptr;
	fopen_s(&fp, filename, "wb");
	if (fp) {
		fwrite(img.data, img.rows * img.cols, 1, fp);
		fclose(fp);
	}
}

void dump_uint8_cwh(const char* filename, TensorShape shape, int minc, int maxc)
{
	FILE* fp = nullptr;
	fopen_s(&fp, filename, "wt");
	if (fp) {
		const int* ptr = shape.iptr;
		int i, j, k = 0;
		for (i = 0; i < shape.wh; i++, ptr++) {
			const int* cptr = ptr;
			for (j = 0; j < shape.c; j++, cptr += shape.wh) {
				if (j >= minc && j < maxc) {
					fprintf(fp, "%02X ", (unsigned char)*cptr);
					if ((k + 1) % 32 == 0) fprintf(fp, "\n");
					k++;
				}
			}
		}
		fclose(fp);
	}
}


static void test_image()
{
	cv::Mat img = cv::imread(DATA_PATH"headshoulder_320x160.png", 0);
	//dump_image(img, "headshoulder_320x160.bin");

	UpixNetwork netq;
	if (!netq.loadNetwork(MODEL_PATH"HeadShoulder_320x160.network.csv")) assert(0);
	if (!netq.loadWeight(MODEL_PATH"HeadShoulder_320x160.weights.txt")) assert(0);
	if (!netq.loadQuant(MODEL_PATH"HeadShoulder_320x160.maxmin.txt")) assert(0);

	
	run_image_int(&netq, img);
	TargetDecoder dec;
	dec.reset(0.25f);
	dec.decode(netq.output(C2_0_IDX).iptr, netq.output(C40_0_IDX).iptr, netq.output(C2_0_IDX).w, netq.output(C2_0_IDX).h, 4, 8);
	dec.decode(netq.output(C2_1_IDX).iptr, netq.output(C40_1_IDX).iptr, netq.output(C2_1_IDX).w, netq.output(C2_1_IDX).h, 4, 4);
	auto targets = dec.getTargets();

	//dump_uint8_cwh("int8_result/50_p1_pc.txt", netq.layers()[50]->output(), 12, 24);
	//dump_uint8_cwh("int8_result/50_p0_pc.txt", netq.layers()[50]->output(), 0, 12);
	dump_int8_result(netq, "int8_result");
}

static BBox cxcywh2box(int cx, int cy, int w, int h)
{
	BBox box;
	box.x1 = (float)cx - w * 0.5f;
	box.y1 = (float)cy - h * 0.5f;
	box.x2 = (float)cx + w * 0.5f;
	box.y2 = (float)cy + h * 0.5f;
	return box;
}

void test_kalman()
{
	KalmanTracker* kt = KalmanTracker_Create(320, 160);

	BBox box = cxcywh2box(0x8c, 0x52, 0x8d, 0x89);//{0.0f, 0.0f, 30.0f, 30.0f};
	KalmanBoxTracker kbt(box);

	KalmanTracker_Reset(kt, &box);

	KalmanTracker_Predict(kt, &box);
	box = kbt.predict();

	box = { 1.0f, 1.0f, 31.0f, 31.0f };
	KalmanTracker_Update(kt, &box);
	kbt.update(box);

	KalmanTracker_Predict(kt, &box);
	box = kbt.predict();

	KalmanTracker_Destroy(kt);
}

int KalmanBoxTracker::g_global_id = 0;
int main(int argc, char** argv)
{
	test_kalman();
	//dump_maxmin(
	//	MODEL_PATH"HeadShoulder_320x160.network.csv",
	//	MODEL_PATH"HeadShoulder_320x160.weights.txt",
	//	VAL_IMAGE_PATH,
	//	MODEL_PATH"HeadShoulder_320x160.maxmin.txt"
	//);
	test_image();
	//test_cam640x480();
	return 0;
}