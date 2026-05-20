#include <filesystem>
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

#define C1_160X128_0_IDX 43
#define C40_160X128_0_IDX 48
#define C1_160X128_1_IDX 59
#define C40_160X128_1_IDX 64
static void run_image(UpixNetwork* net, cv::Mat& img, std::vector<Target> &ftargets, std::vector<Target>& itargets)
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
	net->forward((int*)imgInt.data);

	TargetDecoder dec;
	dec.reset(0.35f);
	dec.decode(net->output(C1_160X128_0_IDX).fptr, net->output(C40_160X128_0_IDX).fptr, net->output(C1_160X128_0_IDX).w, net->output(C1_160X128_0_IDX).h, 1, 8);
	dec.decode(net->output(C1_160X128_1_IDX).fptr, net->output(C40_160X128_1_IDX).fptr, net->output(C1_160X128_1_IDX).w, net->output(C1_160X128_1_IDX).h, 1, 4);
	ftargets = dec.getTargets();
	dec.reset(0.35f);
	dec.decode(net->output(C1_160X128_0_IDX).iptr, net->output(C40_160X128_0_IDX).iptr, net->output(C1_160X128_0_IDX).w, net->output(C1_160X128_0_IDX).h, 1, 8);
	dec.decode(net->output(C1_160X128_1_IDX).iptr, net->output(C40_160X128_1_IDX).iptr, net->output(C1_160X128_1_IDX).w, net->output(C1_160X128_1_IDX).h, 1, 4);
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
	if (net.loadNetwork(MODEL_PATH"128x160.network.csv")
		&& net.loadWeight(MODEL_PATH"128x160.weights.txt"))
	{
		net.quant16();
		cv::VideoCapture vc;
		if (vc.open(0)) {
			cv::Mat frame, gray;			
			cv::Mat input;
			cv::Rect roi(128, 0, 384, 480);
			while (vc.read(frame))
			{
				cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
				cv::resize(gray(roi), input, cv::Size(128, 160));
				
				std::vector<Target> ftargets, itargets;
				run_image(&net, input, ftargets, itargets);
				cv::Mat img_draw = frame(roi);

				cv::rectangle(frame, roi, CV_RGB(0, 0, 0), 2);
				draw_targets(ftargets, img_draw, 3, 0);
				draw_targets(itargets, img_draw, 3, 2);

				cv::imshow("cam", frame);
				int key = cv::waitKey(1);
				if (key == 27) break;
			}
		}
	}
	cv::destroyAllWindows();
	system("pause");
}

static void dump_img()
{
	cv::Mat img = cv::imread(DATA_PATH"128x160.png", 0);

	FILE* fp = nullptr;
	fopen_s(&fp, "128x160.bin", "wb");
	fwrite(img.data, img.cols, img.rows, fp);
	fclose(fp);
}

static void dump_img(const char* name, const cv::Mat& img)
{
	FILE* fp = nullptr;
	fopen_s(&fp, name, "wb");
	fwrite(img.data, img.cols, img.rows, fp);
	fclose(fp);
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

static void dump_int16_result(UpixNetwork& netq, const char* outpath)
{
	std::filesystem::path p(outpath);
	for (auto layer : netq.layers()) {
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

static void test_img_128x160()
{
	cv::Mat img = cv::imread(DATA_PATH"128x160.png");
	UpixNetwork net;
	if (net.loadNetwork(MODEL_PATH"128x160.network.csv")
		&& net.loadWeight(MODEL_PATH"128x160.weights.txt"))
	{
		net.quant16();
		cv::Mat input;
		cv::cvtColor(img, input, cv::COLOR_BGR2GRAY);

		std::vector<Target> ftargets, itargets;
		run_image(&net, input, ftargets, itargets);
		draw_targets(ftargets, img, 1, 0);
		dump_int16_result(net, "int16_res");
	}
}

static std::vector<Target> decode_from_img(const unsigned char* data)
{
	std::vector<Target> targets;
	const int* ptr = (int*)data;
	int count = ptr[0];
	ptr += 1;
	if (count >= 0 && count < 16) {
		for (int i = 0; i < count; i++) {
			Target tar;
			tar.type = ptr[0];
			tar.score = (float)ptr[1];
			tar.w = (float)ptr[2];
			tar.h = (float)ptr[3];
			tar.x = (float)ptr[4];
			tar.y = (float)ptr[5];
			targets.push_back(tar);
			ptr += 6;
		}
	}
	return targets;
}

static void test_u31_cam(int camIdx, const char* networkpath, const char* weightpath)
{
	UpixNetwork net;
	if (net.loadNetwork(networkpath)
		&& net.loadWeight(weightpath))
	{
		net.quant16();

		cv::VideoCapture vc;
		if (vc.open(camIdx)) {
			vc.set(CV_CAP_PROP_FRAME_WIDTH, 320);
			vc.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
			vc.set(CV_CAP_PROP_CONVERT_RGB, 0);

			cv::Mat imgShow(160, 120*2, CV_8UC3);
			cv::Mat img0 = imgShow(cv::Rect(0, 0, 120, 160));
			cv::Mat img1 = imgShow(cv::Rect(120, 0, 120, 160));
			//cv::Mat img2 = imgShow(cv::Rect(0, 160, 120, 160));
			//cv::Mat img3 = imgShow(cv::Rect(120, 160, 120, 160));

			cv::Mat frame;
			cv::Mat img(160, 120, CV_8UC1);
			cv::Mat img_input(160, 128, CV_8UC1);
			cv::Mat input_roi = img_input(cv::Rect(4, 0, 120, 160));
			int frame_count = 0;
			clock_t start_time = clock();


			while (vc.read(frame)) {
				memcpy(img.data, frame.data, 120 * 160);

				cv::cvtColor(img, img0, cv::COLOR_GRAY2BGR);
				img.copyTo(input_roi);

				img0.copyTo(img1);
				//img0.copyTo(img2);
				//img0.copyTo(img3);
				std::vector<Target> ftar, itar;
				run_image(&net, img_input, ftar, itar);
				auto u31_targets = decode_from_img(img.data);
				draw_targets(itar, img0, 1, 0);
				draw_targets(u31_targets, img1, 1, 0);

				//cv::resize(img, imgShow, cv::Size(120 * 2, 160 * 2));
				cv::imshow("u31", imgShow);
				int key = cv::waitKey(1);
				if (key == 27) break;

				frame_count++;
				clock_t now = clock();
				if (now - start_time > 1000) {
					float fps = frame_count * 1000.0f / (now - start_time);
					start_time = now;
					frame_count = 0;
					printf("FPS:%g\r", fps);
				}
			}
		}
		else {
			printf("打开相机失败!\n");
		}
	}
	else {
		printf("没找到模型文件!\n");
	}
}

static void load_txt(const char* filename, cv::Mat& img)
{
	FILE* fp = nullptr;
	fopen_s(&fp, filename, "rt");
	if (fp) {
		unsigned char* ptr = img.data;
		int v;
		while (fscanf_s(fp, "%02X ", &v) == 1) {
			*ptr = (unsigned char)v;
			ptr++;
		}
		fclose(fp);
	}
}

static void test_txt()
{
	cv::Mat img(160, 128, CV_8UC1);
	load_txt("128x160.txt", img);

	UpixNetwork net;
	if (net.loadNetwork(MODEL_PATH"128x160.network.csv")
		&& net.loadWeight(MODEL_PATH"128x160.weights.txt"))
	{
		net.quant16();

		std::vector<Target> ftargets, itargets;
		run_image(&net, img, ftargets, itargets);

		dump_img("128x160-0.bin", img);
	}
}

int main(int argc, char** argv)
{
	int camIdx = 0;
	const char* modelpath = MODEL_PATH"128x160.network.csv";
	const char* weightspath = MODEL_PATH"128x160.weights.txt";
	if (argc == 4) {
		camIdx = atoi(argv[1]);
		modelpath = argv[2];
		weightspath = argv[3];
	}

	//test_txt();
	test_u31_cam(camIdx, modelpath, weightspath);
	//test_img_128x160();
	//dump_img();
	//test_cam640x480();
	return 0;
}