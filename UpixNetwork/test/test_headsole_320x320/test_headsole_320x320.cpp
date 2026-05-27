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
#define DEFAULT_VIDEO "D:/data/UPIX_Raw_2026-05-07_15-17-17.mp4"
#define DEFAULT_OUTPUT "D:/data/UPIX_Raw_2026-05-07_15-17-17_result.mp4"

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

static void test_video(const char* video_name, const char* output_name)
{
	const char* network_name = MODEL_DIR"best.network.csv";
	const char* weights_name = MODEL_DIR"best.weights.txt";
	const char* maxmin_name  = MODEL_DIR"best.maxmin.txt";

	UpixNetwork net;
	if (!net.loadNetwork(network_name)) assert(0);
	if (!net.loadWeight(weights_name)) assert(0);

	UpixNetwork netq;
	if (!netq.loadNetwork(network_name)) assert(0);
	if (!netq.loadWeight(weights_name)) assert(0);
	if (!netq.loadQuant(maxmin_name)) assert(0);

	cv::VideoCapture cap(video_name);
	if (!cap.isOpened()) {
		printf("load video failed: %s\n", video_name);
		return;
	}

	double fps    = cap.get(cv::CAP_PROP_FPS);
	int    width  = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
	int    height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
	int    total  = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);
	printf("video: %s  %dx%d  %.1f fps  %d frames\n", video_name, width, height, fps, total);

	// 输出视频写入器
	cv::VideoWriter writer;
	if (output_name && output_name[0] != '\0') {
		int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
		writer.open(output_name, fourcc, fps, cv::Size(INPUT_W, INPUT_H));
		if (!writer.isOpened()) {
			printf("warning: cannot open output video: %s\n", output_name);
		}
	}

	cv::Mat frame, gray;
	int frame_idx = 0;
	while (true) {
		if (!cap.read(frame) || frame.empty()) break;
		frame_idx++;

		// 转灰度并缩放到网络输入尺寸
		if (frame.channels() == 3)
			cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
		else
			gray = frame.clone();
		cv::resize(gray, gray, cv::Size(INPUT_W, INPUT_H));

		// 推理
		std::vector<Target> ftargets, itargets;
		run_image(&net, &netq, gray, ftargets, itargets);

		// 绘制结果
		cv::Mat draw;
		cv::cvtColor(gray, draw, cv::COLOR_GRAY2BGR);
		draw_targets(itargets, draw, 0);

		// 显示帧号
		char info[64];
		snprintf(info, sizeof(info), "frame %d/%d", frame_idx, total);
		cv::putText(draw, info, cv::Point(4, 16), cv::FONT_HERSHEY_SIMPLEX, 0.45, CV_RGB(0xff, 0xff, 0x00), 1);

		cv::imshow("head_sole_320x320", draw);

		if (writer.isOpened())
			writer.write(draw);

		// 按 'q' 或 ESC 退出
		int key = cv::waitKey(1);
		if (key == 'q' || key == 27) break;
	}

	cap.release();
	if (writer.isOpened()) {
		writer.release();
		printf("result saved to: %s\n", output_name);
	}
	cv::destroyAllWindows();
}

int main(int argc, char** argv)
{
	const char* video_name  = argc > 1 ? argv[1] : DEFAULT_VIDEO;
	const char* output_name = argc > 2 ? argv[2] : DEFAULT_OUTPUT;
	test_video(video_name, output_name);
	return 0;
}
