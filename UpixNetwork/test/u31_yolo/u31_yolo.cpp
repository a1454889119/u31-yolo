#include <time.h>
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

static const cv::Scalar COLORS[] = {
	CV_RGB(0x00, 0xff, 0x00),
	CV_RGB(0xff, 0x00, 0x00),
	CV_RGB(0x00, 0x80, 0x00),
	CV_RGB(0x80, 0x00, 0x00),
	CV_RGB(0x00, 0xc0, 0x40),
	CV_RGB(0xc0, 0x40, 0x00),
};

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
	dec.decode(net->output(63).fptr, net->output(68).fptr, net->output(63).w, net->output(63).h, 16);
	dec.decode(net->output(89).fptr, net->output(94).fptr, net->output(89).w, net->output(89).h, 8);
	dec.decode(net->output(117).fptr, net->output(122).fptr, net->output(117).w, net->output(117).h, 4);
	printf("-----float result\n");
	for (auto& tar : dec.getTargets()) {
		cv::Rect r(
			(int)(tar.x - tar.w * 0.5f),
			(int)(tar.y - tar.h * 0.5f),
			(int)tar.w,
			(int)tar.h);
		if (img.rows == 240) {
			r.y += 40;
		}
		cv::rectangle(img, r, COLORS[tar.type + color_idx]);
		printf("type%d, score%g\n", tar.type, tar.score);
	}
}

static void draw_rects_int(UpixNetwork* net, cv::Mat& img, int color_idx)
{
	TargetDecoder dec;
	dec.reset(0.25f);
	dec.decode(net->output(63).iptr, net->output(68).iptr, net->output(63).w, net->output(63).h, 16);
	dec.decode(net->output(89).iptr, net->output(94).iptr, net->output(89).w, net->output(89).h, 8);
	dec.decode(net->output(117).iptr, net->output(122).iptr, net->output(117).w, net->output(117).h, 4);
	printf("-----int result\n");
	for (auto& tar : dec.getTargets()) {
		cv::Rect r(
			(int)(tar.x - tar.w * 0.5f),
			(int)(tar.y - tar.h * 0.5f),
			(int)tar.w,
			(int)tar.h);
		if (img.rows == 240) {
			r.y += 40;
		}
		cv::rectangle(img, r, COLORS[tar.type + color_idx]);
		printf("type%d, score%g\n", tar.type, tar.score);
	}
}

static void draw_rects_u31(const std::vector<Target>& targets, cv::Mat& img, int color_idx)
{
	printf("-----u31 result\n");
	for (auto& tar : targets) {
		cv::Rect r(
			(int)(tar.x - tar.w * 0.5f),
			(int)(tar.y - tar.h * 0.5f),
			(int)tar.w,
			(int)tar.h);
		if (img.rows == 240) {
			r.y += 40;
		}
		cv::rectangle(img, r, COLORS[tar.type + color_idx]);
		printf("type%d, score%g\n", tar.type, tar.score);
	}

}


static void get_img_data_c0_c2(const unsigned char* input, unsigned char* output, int wh)
{
	for (int i = 0; i < wh; i++, input += 3, output += 2)
	{
		output[0] = input[0];
		output[1] = input[2];
	}
}


static void load_txt_data(const char* filename, int count, unsigned char* data)
{
	FILE* fp = nullptr;
	fopen_s(&fp, filename, "rt");
	if (fp) {
		int v, idx = 0;
		unsigned char* pdata = data;
		while (true)
		{
			if (fscanf_s(fp, "%02x ", &v) == 1) {
				*pdata = (unsigned char)v;
				pdata++;
				idx++;
				if (idx % 32 == 0) {
					fscanf_s(fp, "\n");
				}
				if (idx >= count) break;
			}
			else {
				break;
			}
		}

		fclose(fp);
	}
}



static std::vector<Target> decode_frame(const unsigned char* imgdata)
{
	std::vector<Target> targets;
	const int* iptr = (const int*)imgdata;
	int count = iptr[0];
	iptr++;
	for (int i = 0; i < count; i++) {
		targets.push_back({
			iptr[0],
			iptr[1] / 1024.0f,
			(float)iptr[4],
			(float)iptr[5],
			(float)iptr[2],
			(float)iptr[3],
			});
		iptr += 6;
	}
	return targets;
}

int main(int argc, char** argv)
{
	cv::Mat img1(320, 160, CV_8UC1);
	load_txt_data("160x320.txt", 320 * 160, img1.data);


	UpixNetwork net;
	if (net.loadNetwork(MODEL_PATH"320x160_facepalm.network.csv")
		&& net.loadWeight(MODEL_PATH"320x160_facepalm.weights.txt"))
	{
		UpixNetwork netq;
		if (!netq.loadNetwork(MODEL_PATH"320x160_facepalm.network.csv")) assert(0);
		if (!netq.loadWeight(MODEL_PATH"320x160_facepalm.weights.txt")) assert(0);
		if (!netq.loadQuant(MODEL_PATH"320x160_maxmin.txt")) assert(0);

		cv::VideoCapture vc;
		if (vc.open(1)) {
			vc.set(CV_CAP_PROP_FRAME_WIDTH, 320);
			vc.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
			vc.set(CV_CAP_PROP_CONVERT_RGB, 0);

			cv::Mat imgShow(160 * 2, 320, CV_8UC3);
			cv::Mat img0 = imgShow(cv::Rect(0, 0, 320, 160));
			cv::Mat img1 = imgShow(cv::Rect(0, 160, 320, 160));
			//cv::Mat img2 = imgShow(cv::Rect(0, 320, 320, 160));
			
			cv::Mat frame;
			while (vc.read(frame)) {
				cv::Mat img(320, 160, CV_8UC1);
				memcpy(img.data, frame.data, 160 * 320);
				//get_img_data_c0_c2(frame.data, img.data, 80 * 320);
				cv::transpose(img, img);

				cv::cvtColor(img, img0, cv::COLOR_GRAY2BGR);

				img0.copyTo(img1);
				//img0.copyTo(img2);
				//auto targets = decode_frame(frame.data);

				run_image(&net, img);
				run_image_int(&netq, img);
				draw_rects(&net, img0, 0);
				draw_rects_int(&netq, img1, 0);
				//draw_rects_u31(targets, img2, 0);

				cv::imshow("u31_img", imgShow);
				int key = cv::waitKey(1);
				if (key == 27) break;
			}
		}
	}
	cvDestroyAllWindows();
	system("pause");
}