#include <filesystem>
#include <opencv2/opencv.hpp>
#include <UpixNetwork/Layers/UpixBaseLayer.h>
#include <UpixNetwork/UpixNetwork.h>
#include <UpixNetwork/TargetDecoder.h>
#include <load_output_names.hpp>

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

#define  C2_0_IDX 63
#define C40_0_IDX 68
#define  C2_1_IDX 89
#define C40_1_IDX 94
#define  C2_2_IDX 117
#define C40_2_IDX 122


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
	if (fdata.size() != layer->output().whc) {
		printf("layer %3d size NOT same!!!!\n", layer->info()->idx);
		return;
	}
	stDiff diff = compare_data(fdata, layer->output().fptr);
	printf("layer %3d, avgdiff:%g maxdiff:%g, value:%g, my value:%g\n",
		layer->info()->idx, diff.avg, diff.maxv, fdata[diff.idx], layer->output().fptr[diff.idx]);
}

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
	dec.decode(net->output(C2_0_IDX).fptr, net->output(C40_0_IDX).fptr, net->output(C2_0_IDX).w, net->output(C2_0_IDX).h, 2, 16);
	dec.decode(net->output(C2_1_IDX).fptr, net->output(C40_1_IDX).fptr, net->output(C2_1_IDX).w, net->output(C2_1_IDX).h, 2, 8);
	dec.decode(net->output(C2_2_IDX).fptr, net->output(C40_2_IDX).fptr, net->output(C2_2_IDX).w, net->output(C2_2_IDX).h, 2, 4);
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
	dec.decode(net->output(C2_0_IDX).iptr, net->output(C40_0_IDX).iptr, net->output(C2_0_IDX).w, net->output(C2_0_IDX).h, 2, 16);
	dec.decode(net->output(C2_1_IDX).iptr, net->output(C40_1_IDX).iptr, net->output(C2_1_IDX).w, net->output(C2_1_IDX).h, 2, 8);
	dec.decode(net->output(C2_2_IDX).iptr, net->output(C40_2_IDX).iptr, net->output(C2_2_IDX).w, net->output(C2_2_IDX).h, 2, 4);
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
		printf("type%d, score%x, w:%x, h:%x, x:%x, y:%x \n", tar.type, (int)(tar.score*1024.0f), 
			(int)(tar.w), 
			(int)(tar.h), 
			(int)(tar.x), 
			(int)(tar.y));
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


static void dump_maxmin(UpixNetwork& net, const char* imgpath)
{
	auto names = get_img_full_names(imgpath);

	std::vector<stMaxMin> mms(net.layers().size());
	for (auto& name : names) {
		printf("%s                   \r", name.c_str());
		cv::Mat img = cv::imread(name.c_str());
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
		printf("layer%d maxv:%g minv:%g\n", i, mm.maxv, mm.minv);
	}
}

static std::string replace(const std::string& src, char old, char _new)
{
	std::string str = src;
	std::string::size_type pos = src.find(old);
	while (pos != std::string::npos) {
		str[pos] = _new;
		pos = str.find(old);
	}
	return str;

}

static void rename_outputs(const char* output_path, const char* output_names_file)
{
	std::filesystem::path p(output_path);
	auto output_names = load_output_names(output_names_file);
	for (int i = 0; i < (int)output_names.size(); i++) {
		output_names[i] = replace(output_names[i], '/', '_');
		auto pp = p / (output_names[i] + ".txt");
		std::filesystem::rename(pp, p / (std::to_string(i) + ".txt"));		
	}
}

static void runImages(const char* imgpath, UpixNetwork& net, UpixNetwork& netq)
{
	auto names = get_img_full_names(imgpath);
	for (auto& name : names) {
		cv::Mat img = cv::imread(name);
		printf("%s\n", name.c_str());
		run_image(&net, img);
		run_image_int(&netq, img);
		draw_rects(&net, img, 0);
		draw_rects_int(&netq, img, 2);
		cv::imshow("runImage", img);
		int key = cv::waitKey(0);
		if (key == 27) break;
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

static void dump_bin(const char* input_png, const char* output_bin)
{
	cv::Mat img = cv::imread(input_png, 0);
	FILE* fp = nullptr;
	fopen_s(&fp, output_bin, "wb");
	if (fp) {
		fwrite(img.data, img.cols, img.rows, fp);
		fclose(fp);
	}
}


static void draw_rects_640(UpixNetwork* net, cv::Mat& img, int color_idx)
{
	TargetDecoder dec;
	dec.reset(0.35f);
	dec.decode(net->output(C2_0_IDX).fptr, net->output(C40_0_IDX).fptr, net->output(C2_0_IDX).w, net->output(C2_0_IDX).h, 2, 16);
	dec.decode(net->output(C2_1_IDX).fptr, net->output(C40_1_IDX).fptr, net->output(C2_1_IDX).w, net->output(C2_1_IDX).h, 2, 8);
	dec.decode(net->output(C2_2_IDX).fptr, net->output(C40_2_IDX).fptr, net->output(C2_2_IDX).w, net->output(C2_2_IDX).h, 2, 4);
	printf("-----float result\n");
	for (auto& tar : dec.getTargets()) {
		cv::Rect r(
			(int)(tar.x - tar.w * 0.5f) * 2,
			(int)(tar.y - tar.h * 0.5f) * 2 + 80,
			(int)tar.w * 2,
			(int)tar.h * 2);
		
		cv::rectangle(img, r, COLORS[tar.type + color_idx]);
		printf("type%d, score%g\n", tar.type, tar.score);
	}
}

static void draw_rects_640_int(UpixNetwork* net, cv::Mat& img, int color_idx)
{
	TargetDecoder dec;
	dec.reset(0.35f);
	dec.decode(net->output(C2_0_IDX).iptr, net->output(C40_0_IDX).iptr, net->output(C2_0_IDX).w, net->output(C2_0_IDX).h, 2, 16);
	dec.decode(net->output(C2_1_IDX).iptr, net->output(C40_1_IDX).iptr, net->output(C2_1_IDX).w, net->output(C2_1_IDX).h, 2, 8);
	dec.decode(net->output(C2_2_IDX).iptr, net->output(C40_2_IDX).iptr, net->output(C2_2_IDX).w, net->output(C2_2_IDX).h, 2, 4);
	printf("-----int result\n");
	for (auto& tar : dec.getTargets()) {
		cv::Rect r(
			(int)(tar.x - tar.w * 0.5f) * 2,
			(int)(tar.y - tar.h * 0.5f) * 2 + 80,
			(int)tar.w * 2,
			(int)tar.h * 2);

		cv::rectangle(img, r, COLORS[tar.type + color_idx]);
		printf("type%d, score%g\n", tar.type, tar.score);
	}
}

static void test_cam()
{
	UpixNetwork net;
	if (net.loadNetwork(MODEL_PATH"320x160_facepalm.network.csv")
		&& net.loadWeight(MODEL_PATH"320x160_facepalm.weights.txt"))
	{
		UpixNetwork netq;
		if (!netq.loadNetwork(MODEL_PATH"320x160_facepalm.network.csv")) assert(0);
		if (!netq.loadWeight(MODEL_PATH"320x160_facepalm.weights.txt")) assert(0);
		if (!netq.loadQuant(MODEL_PATH"320x160_maxmin.txt")) assert(0);

		cv::VideoCapture vc;
		if (vc.open(0)) {
			cv::Mat frame, gray;
			while (vc.read(frame))
			{
				cv::Mat img0 = frame(cv::Rect(0, 80, 640, 320));
				cv::cvtColor(img0, gray, cv::COLOR_BGR2GRAY);
				cv::resize(gray, gray, cv::Size(320, 160));

				run_image(&net, gray);
				//run_image(&netq, img);
				run_image_int(&netq, gray);

				draw_rects_640(&net, frame, 0);
				//draw_rects(&netq, img, 2);
				draw_rects_640_int(&netq, frame, 4);

				cv::imshow("cam", frame);
				int key = cv::waitKey(1);
				if (key == 27) break;
			}
		}

	}
	cv::destroyAllWindows();
	system("pause");
}

#define C1_160X128_0_IDX 43
#define C40_160X128_0_IDX 48
#define C1_160X128_1_IDX 59
#define C40_160X128_1_IDX 64
static void draw_rects_640_128(UpixNetwork* net, cv::Mat& img, int color_idx)
{
	TargetDecoder dec;
	dec.reset(0.25f);	
	dec.decode(net->output(C1_160X128_0_IDX).fptr, net->output(C40_160X128_0_IDX).fptr, net->output(C1_160X128_0_IDX).w, net->output(C1_160X128_0_IDX).h, 1, 8);
	dec.decode(net->output(C1_160X128_1_IDX).fptr, net->output(C40_160X128_1_IDX).fptr, net->output(C1_160X128_1_IDX).w, net->output(C1_160X128_1_IDX).h, 1, 4);
	printf("-----float result\n");
	for (auto& tar : dec.getTargets()) {
		cv::Rect r(
			(int)(tar.x - tar.w * 0.5f) * 4,
			(int)(tar.y - tar.h * 0.5f) * 4 - 16,
			(int)tar.w * 4,
			(int)tar.h * 4);

		cv::rectangle(img, r, COLORS[tar.type + color_idx]);
		printf("type%d, score%g\n", tar.type, tar.score);
	}
}

static void draw_rects_int_640_128(UpixNetwork* net, cv::Mat& img, int color_idx)
{
	TargetDecoder dec;
	dec.reset(0.25f);
	dec.decode(net->output(C1_160X128_0_IDX).iptr, net->output(C40_160X128_0_IDX).iptr, net->output(C1_160X128_0_IDX).w, net->output(C1_160X128_0_IDX).h, 1, 8);
	dec.decode(net->output(C1_160X128_1_IDX).iptr, net->output(C40_160X128_1_IDX).iptr, net->output(C1_160X128_1_IDX).w, net->output(C1_160X128_1_IDX).h, 1, 4);
	printf("-----int result\n");
	for (auto& tar : dec.getTargets()) {
		cv::Rect r(
			(int)(tar.x - tar.w * 0.5f) * 4,
			(int)(tar.y - tar.h * 0.5f) * 4 - 16,
			(int)tar.w * 4,
			(int)tar.h * 4);

		cv::rectangle(img, r, COLORS[tar.type + color_idx]);
		printf("type%d, score%g\n", tar.type, tar.score);
	}
}



static void test_cam_160x128()
{
	UpixNetwork net;
	if (net.loadNetwork(MODEL_PATH"160x128.network.csv")
		&& net.loadWeight(MODEL_PATH"160x128.weights.txt"))
	{
		net.quant16();
		cv::VideoCapture vc;
		if (vc.open(0)) {
			cv::Mat frame;
			cv::Mat gray_640x512(512, 640, CV_8UC1);
			gray_640x512 = 0;
			cv::Mat gray_640x480 = gray_640x512(cv::Rect(0, 16, 640, 480));
			cv::Mat input;
			while (vc.read(frame))
			{
				cv::cvtColor(frame, gray_640x480, cv::COLOR_BGR2GRAY);
				cv::resize(gray_640x512, input, cv::Size(160, 128));

				run_image(&net, input);
				draw_rects_640_128(&net, frame, 0);

				run_image_int(&net, input);
				draw_rects_int_640_128(&net, frame, 2);

				cv::imshow("cam", frame);
				int key = cv::waitKey(1);
				if (key == 27) break;
			}
		}
	}
	cv::destroyAllWindows();
	system("pause");
}

static void test_pool_320x160()
{
	cv::Mat img = cv::imread(DATA_PATH"facepalm.png");
	UpixNetwork net;
	if (net.loadNetwork(MODEL_PATH"facepalm_320x160_3types.network.csv")
		&& net.loadWeight(MODEL_PATH"facepalm_320x160_3types.weights.txt"))
	{
		UpixNetwork netq;
		if (!netq.loadNetwork(MODEL_PATH"facepalm_320x160_3types.network.csv")) assert(0);
		if (!netq.loadWeight(MODEL_PATH"facepalm_320x160_3types.weights.txt")) assert(0);
		if (!netq.loadQuant(MODEL_PATH"facepalm_320x160_3types.maxmin.txt")) assert(0);
		//netq.dumpQuantedWeights("320x160_quanted_weights.txt");
		//dump_maxmin(net, "D:\\git\\u31-yolo\\yolo-c\\data\\val_UPIX_face_human_images_0915_person\\");
		//runImages(TEST_IMAGES_PATH, net, netq);

		cv::resize(img, img, cv::Size(320, 160));
		cv::Mat input;
		cv::cvtColor(img, input, cv::COLOR_BGR2GRAY);
		

		run_image(&net, input);
		//run_image(&netq, img);
		run_image_int(&netq, input);

		draw_rects(&net, img, 0);
		////draw_rects(&netq, img, 2);
		draw_rects_int(&netq, img, 4);
		//dump_int_result(netq, "output_int");
		
		//for (auto layer : net.layers()) {
		//	std::string name = DATA_PATH"float_res/1/" + std::to_string(layer->info()->idx) + ".txt";
		//
		//	check_layer_output(layer, name.c_str());
		//}
	}

}




#define TEST_IMAGES_PATH "D:\\git\\u31-yolo\\yolo-c\\data\\val_UPIX_face_human_images_0915_person\\"
int main(int argc, char** argv)
{
	test_pool_320x160();
	//test_cam_160x128();
	//test_cam();
	//dump_bin(DATA_PATH"facepalm.png", "facepalm_320x160.bin");
//	dump_bin(DATA_PATH"000.png", "320x160-000.bin");
//	cv::Mat img0(320, 160, CV_8UC1);
//	cv::Mat img1(160, 320, CV_8UC1);
//
//	load_txt_data("320x160.txt", 320 * 160, img1.data);
//	load_txt_data("160x320.txt", 320 * 160, img0.data);

	//rename_outputs(DATA_PATH"float_res/1/", DATA_PATH"320x160_output_names.txt");
	//UpixNetwork net;
	//if (net.loadNetwork(MODEL_PATH"320x160.network.csv")
	//	&& net.loadWeight(MODEL_PATH"320x160.weights.txt"))
	//{
	//	UpixNetwork netq;
	//	if (!netq.loadNetwork(MODEL_PATH"320x160.network.csv")) assert(0);
	//	if (!netq.loadWeight(MODEL_PATH"320x160.weights.txt")) assert(0);
	//	if (!netq.loadQuant(MODEL_PATH"320x160_maxmin.txt")) assert(0);
	//	//netq.dumpQuantedWeights("320x160_quanted_weights.txt");
	//	//dump_maxmin(net, "D:\\git\\u31-yolo\\yolo-c\\data\\val_UPIX_face_human_images_0915_person\\");
	//	//runImages(TEST_IMAGES_PATH, net, netq);
	//
	//	cv::Mat img = cv::imread(DATA_PATH"000.png");
	//	run_image(&net, img);
	//	//run_image(&netq, img);
	//	run_image_int(&netq, img);
	//	draw_rects(&net, img, 0);
	//	//draw_rects(&netq, img, 2);
	//	draw_rects_int(&netq, img, 4);
	//	//dump_int_result(netq, "output_int");
	//	
	//	//for (auto layer : net.layers()) {
	//	//	std::string name = DATA_PATH"float_res/1/" + std::to_string(layer->info()->idx) + ".txt";
	//	//
	//	//	check_layer_output(layer, name.c_str());
	//	//}
	//}
	cv::destroyAllWindows();
	system("pause");
	return 0;
}