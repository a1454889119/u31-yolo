#include <stdlib.h>
#include <string.h>
#include <fstream>
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

static short* get_input_data(const cv::Mat& img)
{
	int wh = img.rows * img.cols;
	const unsigned char* src_ptr = img.data;
	short* input_data = (short*)malloc(sizeof(short) * wh * img.channels());
	if (!input_data) return nullptr;

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
	
	return input_data;
}

#define PIXEL_FORMAT_Y 0
#define PIXEL_FORMAT_RGB 1
#define PIXEL_FORMAT_BGR 2


struct stResultContent {
	std::string name;
	std::vector<unsigned char> data;
};

struct stResultRects {
	std::string name;
	std::vector<TargetRect> rects;
};

static bool compare_rects(
	const std::vector<TargetRect>& rects0,
	const std::vector<TargetRect>& rects1,
	std::string& text)
{
	
	text = "";
	bool same = true;
	if (rects0.size() != rects1.size()) {
		text += "目标数量不同!!!\n";
		same = false;
	}
	else {
		int i;
		text += "" + std::to_string(rects0.size()) + "个目标\n";
		for (i = 0; i < rects0.size(); i++) {
			auto& r0 = rects0[i];
			auto& r1 = rects1[i];
			if (memcmp(&r0, &r1, sizeof(r0)) != 0) {
				same = false;
				text += "  目标" + std::to_string(i) + " ";
				if (r0.type != r1.type) {
					text += "类型不同!";
				}
				else {
					if (r0.score != r1.score) {
						text += "得分差" + std::to_string(r0.score - r1.score) + ",";
					}
					if (r0.x != r1.x) {
						text += "x差" + std::to_string(r0.x - r1.x) + ",";
					}
					if (r0.y != r1.y) {
						text += "y差" + std::to_string(r0.y - r1.y) + ",";
					}
					if (r0.w != r1.w) {
						text += "w差" + std::to_string(r0.w - r1.w) + ",";
					}
					if (r0.h != r1.h) {
						text += "h差" + std::to_string(r0.h - r1.h) + ",";
					}
				}
				text += "\n";
			}
		}
	}
	if (same) {
		text += "完全相同\n";
	}
	return same;
}


static void detect_image(S16Network* network, const char* img_name, int fmt, const stResultRects& u31_res)
{
	static const cv::Scalar colors[] = {
		CV_RGB(0x00, 0xff, 0x00),
		CV_RGB(0xff, 0x00, 0x00),
		CV_RGB(0x00, 0x00, 0xff),
		CV_RGB(0x80, 0x80, 0x00),
		CV_RGB(0x00, 0x80, 0x80),
		CV_RGB(0x80, 0x00, 0x80),
		CV_RGB(0xff, 0x00, 0x80),
		CV_RGB(0x40, 0xff, 0x40),
	};

	std::vector<TargetRect> result;
	cv::Mat img = cv::imread(img_name), img_show;
	switch (fmt) {
	case PIXEL_FORMAT_Y: 
		cv::cvtColor(img, img, CV_BGR2GRAY); 
		break;
	case PIXEL_FORMAT_RGB: 
		cv::cvtColor(img, img, CV_BGR2RGB); 
		break;
	case PIXEL_FORMAT_BGR: 		
		break;
	}
	img_show = img.clone();
	if (img.rows != network->inputH() || img.cols != network->inputW()) {
		cv::resize(img, img, cv::Size(network->inputW(), network->inputH()));
		cv::resize(img_show, img_show, cv::Size(network->inputW(), network->inputH()));
	}
	short* input_data = get_input_data(img);

	network->forward((short*)input_data);
	int n = decode_rects(
		network->getLayerOutput(98),
		network->getLayerOutput(100),
		network->getLayerOutput(106),

		network->getLayerOutput(113),
		network->getLayerOutput(115),
		network->getLayerOutput(121)
	);

	const TargetRect* prect = get_decoded_rects();
	int i;
	for (i = 0; i < n; i++, prect++) {
		result.push_back(*prect);
	}
	free(input_data);

	std::string text;
	if (!compare_rects(result, u31_res.rects, text)) {
		printf("%s %s\n", u31_res.name.c_str(), text.c_str());
		cv::Mat img_cmp(img.rows, img.cols * 2, img.type());
		cv::Mat pc_img(img_cmp(cv::Rect(0, 0, network->inputW(), network->inputH())));
		cv::Mat u31_img(img_cmp(cv::Rect(network->inputW(), 0, network->inputW(), network->inputH())));
		img_show.copyTo(pc_img);
		img_show.copyTo(u31_img);

		for (auto& r : result) {
			cv::rectangle(pc_img, cv::Rect(r.x >> 10, r.y >> 10, r.w >> 10, r.h >> 10), colors[r.type]);
		}

		for (auto& r : u31_res.rects) {
			cv::rectangle(u31_img, cv::Rect(r.x >> 10, r.y >> 10, r.w >> 10, r.h >> 10), colors[r.type]);
		}

		cv::imshow("yolo", img_cmp);
		cv::waitKey(0);
	}
	else {
		printf("                                                                     \r");
		printf("%s完全相同!\r", u31_res.name.c_str());
	}
}

static void usage() 
{
	printf("----------\n");
	printf("使用6个参数:\n");
	printf("    arg1: 网络结构文件(csv)\n");
	printf("    arg2: 网络权重文件(txt)\n");
	printf("    arg3: 测试图像路径(文件夹)\n");
	printf("    arg4: 图像分辨率(如320x224)\n");
	printf("    arg5: 像素格式(y/rgb/bgr)\n");
	printf("    arg6: U31结果文件(txt)\n");
}


static bool load_result_file(const char* filename, std::vector<stResultContent>& content)
{
	content.clear();
	FILE* fp = fopen(filename, "rt");
	if (fp) {
		char name[64];
		int v, idx;
		while (true)
		{
			if (fscanf(fp, "%s\n", name) != 1) break;
			idx = 0;
			stResultContent res = { name, {} };
			while (true)
			{
				if (fscanf(fp, "%02x ", &v) == 1) {
					res.data.push_back((unsigned char)v);
					idx++;

					if (idx % 32 == 0) {
						fscanf(fp, "\n");
					}
				}
				else {
					fscanf(fp, "\n\n");
					break;
				}
			}
			content.push_back(res);
		}
		fclose(fp);
	}
	return content.size() > 0;
}

static bool decode_result_file(const char* filename, std::vector<stResultRects>& res)
{
	std::vector<stResultContent> content;
	res.clear();
	if (!load_result_file(filename, content)) {
		printf("读取U31结果文件(txt)错误:%s", filename);
		return false;
	}

	for (auto& c : content) {
		const int* pdata = (const int*)c.data.data();
		int n = (pdata[0] - 0x19000) / 32;
		if (n < 0 || n * 32 + 4 > c.data.size()) {
			printf("%s: 数据错误!\n", c.name.c_str());
			continue;
		}
		stResultRects rr = { c.name, {} };
		pdata++;
		for (int i = 0; i < n; i++, pdata += 8) {
			rr.rects.push_back({ pdata[0], pdata[1], pdata[4], pdata[5], pdata[6], pdata[7] });
		}
		res.push_back(rr);
	}
	
	if (res.empty()) {
		printf("未解析得到任何结果:%s", filename);
		return false;
	}
	return true;
}


static void image2bin(const cv::Mat& img, const char* filename)
{
	FILE* fp = fopen(filename, "wb");
	if (fp) {
		fwrite(img.data, img.cols, img.rows * img.channels(), fp);
		fclose(fp);
	}
}

static void image2upasm(const cv::Mat& img, const char* filename)
{	
	FILE* fp = fopen(filename, "wt");
	if (fp) {
		int c = img.channels();
		int whc = img.rows * img.cols * c;
		unsigned char* ptr = img.data;
		fprintf(fp, "@export IMAGE_DATA\n");
		fprintf(fp, "IMAGE_DATA:\n");
		for (int i = 0; i < whc; i++) {
			if (i % 32 == 0) {
				fprintf(fp, ".byte ");
			}
			fprintf(fp, "0x%02x ", ptr[i]);
			if ((i + 1) % 32 == 0) {
				fprintf(fp, "\n");
			}
		}
		fclose(fp);
	}
}


static void dump_data_whc(const char* filename, const S16Tensor *tensor)
{
	FILE* fp = fopen(filename, "wt");
	if (fp) {
		int i;
		for (i = 0; i < tensor->whc; i++) {
			fprintf(fp, "%04X ", (unsigned short)tensor->ptr[i]);
			if ((i + 1) % 16 == 0) {
				fprintf(fp, "\n");
			}
		}
		fclose(fp);
	}
}

static void dump_data_cwh(const char* filename, const S16Tensor* tensor)
{
	FILE* fp = fopen(filename, "wt");
	if (fp) {
		int i, j, k = 0, wh = tensor->w * tensor->h;
		for (i = 0; i < wh; i++) {
			for (j = 0; j < tensor->c; j++) {
				fprintf(fp, "%04X ", (unsigned short)tensor->ptr[j*wh + i]);
				k++;
				if (k == 16) {
					fprintf(fp, "\n");
					k = 0;
				}
			}			
		}
		fclose(fp);
	}
}

static void dump_upasm_whc(const char* filename, const char*labelname, const S16Tensor* tensor)
{
	FILE* fp = fopen(filename, "wt");
	if (fp) {
		int i;
		fprintf(fp, "@export %s\n", labelname);
		fprintf(fp, ".align 16\n");
		fprintf(fp, "%s:\n", labelname);

		for (i = 0; i < tensor->whc; i++) {
			if (i % 16 == 0) {
				fprintf(fp, ".short ");
			}
			fprintf(fp, "0x%04x ", (unsigned short)tensor->ptr[i]);
			if ((i + 1) % 16 == 0) {
				fprintf(fp, "\n");
			}
		}
		fclose(fp);
	}
}

static void dump_upasm_cwh(const char* filename, const char* labelname, const S16Tensor* tensor)
{
	FILE* fp = fopen(filename, "wt");
	if (fp) {
		fprintf(fp, "@export %s\n", labelname);
		fprintf(fp, ".align 16\n");
		fprintf(fp, "%s:\n", labelname);

		int i, j, k = 0, wh = tensor->w * tensor->h;
		for (i = 0; i < wh; i++) {
			for (j = 0; j < tensor->c; j++) {
				if (k == 0) {
					fprintf(fp, ".short ");
				}
				fprintf(fp, "0x%04x ", (unsigned short)tensor->ptr[j * wh + i]);
				k++;
				if (k == 16) {
					fprintf(fp, "\n");
					k = 0;
				}
			}
		}
		fclose(fp);
	}
}


#define IMAGE_PATH "\\\\192.168.1.173\\优象资料库\\研发部\\研发部\\01-测试数据wdz\\06-YOLO数据集\\Png\\"
#define IMAGE_NAME "pic_20240902_162232_640x480.png"

void test_img()
{
	S16Network* network = new S16Network;
	network->loadNetwork("../data/320x224network.csv", 0);
	network->loadWeight("../data/320x224weight.txt");
	cv::Mat img = cv::imread(IMAGE_PATH IMAGE_NAME);
	cv::resize(img, img, cv::Size(network->inputW(), network->inputH()));
	
	short* input_data = get_input_data(img);
	network->forward(input_data);

	network->dumpLayer(98, 100, "c6_20x14-pc.txt");
	network->dumpLayer(106, "c12_20x14-pc.txt");		
	int n = decode_rects(
		network->getLayerOutput(98),
		network->getLayerOutput(100),
		network->getLayerOutput(106),

		network->getLayerOutput(113),
		network->getLayerOutput(115),
		network->getLayerOutput(121)
	);
	free(input_data);
	delete network;
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

void test_160x120()
{
	std::map<int, std::string> layer_idx_names = {
		{0, "_model.0_act_LeakyRelu_output_0"},
		{1, "_model.1_cv1_act_LeakyRelu_output_0"},
		{2, "_model.1_cv2_conv_Conv_output_0"},
		{3, "_model.2_cv1_act_LeakyRelu_output_0"},
		{5, "_model.2_m.0_dw1_act_LeakyRelu_output_0"},
		{4, "_model.2_m.0_pw1_act_LeakyRelu_output_0"},
		{7, "_model.2_m.0_dw2_act_LeakyRelu_output_0"},
		{6, "_model.2_m.0_pw2_act_LeakyRelu_output_0"},
		{9, "_model.2_m.1_dw1_act_LeakyRelu_output_0"},
		{8, "_model.2_m.1_pw1_act_LeakyRelu_output_0"},
		{11, "_model.2_m.1_dw2_act_LeakyRelu_output_0"},
		{10, "_model.2_m.1_pw2_act_LeakyRelu_output_0"},
		{12, "_model.2_cv2_act_LeakyRelu_output_0"},
		{13, "_model.3_act_LeakyRelu_output_0"},
		{14, "_model.4_act_LeakyRelu_output_0"},
		{15, "_model.5_cv1_act_LeakyRelu_output_0"},
		{17, "_model.5_m.0_dw1_act_LeakyRelu_output_0"},
		{16, "_model.5_m.0_pw1_act_LeakyRelu_output_0"},
		{19, "_model.5_m.0_dw2_act_LeakyRelu_output_0"},
		{18, "_model.5_m.0_pw2_act_LeakyRelu_output_0"},
		{21, "_model.5_m.1_dw1_act_LeakyRelu_output_0"},
		{20, "_model.5_m.1_pw1_act_LeakyRelu_output_0"},
		{23, "_model.5_m.1_dw2_act_LeakyRelu_output_0"},
		{22, "_model.5_m.1_pw2_act_LeakyRelu_output_0"},
		{25, "_model.5_m.2_dw1_act_LeakyRelu_output_0"},
		{24, "_model.5_m.2_pw1_act_LeakyRelu_output_0"},
		{27, "_model.5_m.2_dw2_act_LeakyRelu_output_0"},
		{26, "_model.5_m.2_pw2_act_LeakyRelu_output_0"},
		{28, "_model.5_cv2_act_LeakyRelu_output_0"},
		{29, "_model.6_cv1_act_LeakyRelu_output_0"},
		{30, "_model.6_m_MaxPool_output_0"},
		{31, "_model.6_m_1_MaxPool_output_0"},
		{32, "_model.6_m_2_MaxPool_output_0"},
		{33, "_model.6_cv2_act_LeakyRelu_output_0"},
		{34, "_model.7_act_LeakyRelu_output_0"},
		{35, "_model.8_act_LeakyRelu_output_0"},
		{36, "_model.9_pool_GlobalAveragePool_output_0"},
		{37, "_model.9_act_Sigmoid_output_0"},
		{38, "_model.9_Mul_output_0"},

		{39, "_model.15_cv3.0_cv3.0.0_act_LeakyRelu_output_0"},
		{40, "_model.15_cv3.0_cv3.0.1_conv_Conv_output_0"},
		{41, "_model.15_cv3.0_cv3.0.2_act_LeakyRelu_output_0"},
		{42, "_model.15_cv3.0_cv3.0.3_conv_Conv_output_0"},
		{43, "_model.15_cv3.0_cv3.0.4_Conv_output_0"},
		{44, "_model.15_cv2.0_cv2.0.0_act_LeakyRelu_output_0"},
		{45, "_model.15_cv2.0_cv2.0.1_conv_Conv_output_0"},
		{46, "_model.15_cv2.0_cv2.0.2_act_LeakyRelu_output_0"},
		{47, "_model.15_cv2.0_cv2.0.3_conv_Conv_output_0"},
		{48, "_model.15_cv2.0_cv2.0.4_Conv_output_0"},

		{49, "_model.10_Resize_output_0"},
		{50, "_model.12_act_LeakyRelu_output_0"},
		{51, "_model.13_act_LeakyRelu_output_0"},
		{52, "_model.14_pool_GlobalAveragePool_output_0"},
		{53, "_model.14_act_Sigmoid_output_0"},
		{54, "_model.14_Mul_output_0"},

		{55, "_model.15_cv3.1_cv3.1.0_act_LeakyRelu_output_0"},
		{56, "_model.15_cv3.1_cv3.1.1_conv_Conv_output_0"},
		{57, "_model.15_cv3.1_cv3.1.2_act_LeakyRelu_output_0"},
		{58, "_model.15_cv3.1_cv3.1.3_conv_Conv_output_0"},
		{59, "_model.15_cv3.1_cv3.1.4_Conv_output_0"},
		{60, "_model.15_cv2.1_cv2.1.0_act_LeakyRelu_output_0"},
		{61, "_model.15_cv2.1_cv2.1.1_conv_Conv_output_0"},
		{62, "_model.15_cv2.1_cv2.1.2_act_LeakyRelu_output_0"},
		{63, "_model.15_cv2.1_cv2.1.3_conv_Conv_output_0"},
		{64, "_model.15_cv2.1_cv2.1.4_Conv_output_0"},
	};
	const std::string prefix = "./onnx_intermediate_outputs_single_image_4/";

	S16Network* network = new S16Network;
	network->loadNetwork("../data/160x128.network.csv", 1);
	network->loadWeight("../data/160x128.weights.txt");
	cv::Mat img_show = cv::imread("160x120-1.png", 1);
	cv::Mat img, imgTrans;
	cv::cvtColor(img_show, img, CV_BGR2GRAY);

	//cv::Mat img128(128, 160, CV_8UC1);
	//img128 = 0;
	//img.copyTo(img128(cv::Rect(0, 4, 160, 120)));

	//cv::resize(img, img, cv::Size(160, 128));
	//cv::transpose(img, imgTrans);
	//cv::flip(imgTrans, imgTrans, 0);


	cv::resize(img_show, img_show, cv::Size(640, 512));

	//cv::Mat imgF;
	//img.convertTo(imgF, CV_32FC1);
	//imgF /= 255.0f;
	
	short* input_data = get_input_data(img);
	network->forward(input_data);

//	image2bin(imgTrans, "120x160.bin");
//
//	image2upasm(img, "IMAGE_160x120.upasm");
//	image2bin(img, "160x120-a.bin");
//	const std::string out_prefix = "./160x120_a_s16_result/";
//	for (auto layer : network->layers()) {
//		auto outlabel = "LAYER_" + std::to_string(layer->idx) + "_OUTPUT";
//		auto inlabel = "LAYER_" + std::to_string(layer->idx) + "_INPUT";
//		if (layer->out_whc) {
//			auto txtname = out_prefix + "layer_" + std::to_string(layer->idx) + "_whc.txt";
//			auto upasmname = out_prefix + "layer_" + std::to_string(layer->idx) + "_whc.upasm";
//			dump_data_whc(txtname.c_str(), &layer->output);
//			dump_upasm_whc(upasmname.c_str(), outlabel.c_str(), &layer->output);
//		}
//		else {
//			auto txtname = out_prefix + "layer_" + std::to_string(layer->idx) + "_cwh.txt";
//			auto upasmname = out_prefix + "layer_" + std::to_string(layer->idx) + "_cwh.upasm";
//			dump_data_cwh(txtname.c_str(), &layer->output);
//			dump_upasm_cwh(upasmname.c_str(), outlabel.c_str(), &layer->output);
//		}
//
//		if (layer->type == S16Layer_Convolution && layer->from_count > 1) {
//			if (layer->in_whc) {
//				auto upasmname = out_prefix + "layer_" + std::to_string(layer->idx) + "_input_whc.upasm";
//				dump_upasm_whc(upasmname.c_str(), inlabel.c_str(), &layer->input);
//			}
//			else {
//				auto upasmname = out_prefix + "layer_" + std::to_string(layer->idx) + "_input_cwh.upasm";
//				dump_upasm_cwh(upasmname.c_str(), inlabel.c_str(), &layer->input);
//			}
//		}
//	}


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
	auto targets = get_decoded_rects_160x120();
	cv::Scalar colors[2] = { CV_RGB(0, 255, 0), CV_RGB(0, 0, 255) };
	for (int i = 0; i < n; i++) {
		cv::Rect r;
		r.x = (targets[i].x - targets[i].w / 2) >> 8;
		r.y = (targets[i].y - targets[i].h / 2) >> 8;
		r.width = targets[i].w >> 8;
		r.height = targets[i].h >> 8;
		r.y -= 16;
		cv::rectangle(img_show, r, colors[targets[i].type], 1);
	}

	for (auto layer : network->layers()) {
		auto iter = layer_idx_names.find(layer->idx);
		assert(iter != layer_idx_names.end());
		auto filename = prefix + iter->second + ".txt";
		check_layer_output(layer, filename.c_str());
	}


	free(input_data);
	delete network;
}

static std::vector<std::string> load_float_text(const std::string& filename)
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

struct stTextInfo {
	int layer_idx;
	int c;
	int off;
};

static void set_info_map(
	std::map<std::string, stTextInfo>& infomap,
	const std::vector<std::string>& infos,
	int layer, int wh)
{
	for (size_t i = 0; i < infos.size(); i++) {
		auto& text = infos[i];
		auto iter = infomap.find(text);
		if (iter == infomap.end()) {
			int c = i / wh;
			int off = i % wh;
			infomap.insert({ text, {layer, c, off} });
		}
	}
}

static void check_info_map(
	const std::map<std::string, stTextInfo>& infomap,
	const std::vector<std::string>& infos,
	const std::string& fname)
{
	FILE* fp = fopen(fname.c_str(), "wt");
	if (fp) {
		for (auto& text : infos) {
			auto iter = infomap.find(text);
			if (iter != infomap.end()) {
				const stTextInfo& info = iter->second;
				fprintf(fp, "%s, layer:%2d c:%2d, off:%4d\n", text.c_str(), info.layer_idx, info.c, info.off);
			}
		}
		fclose(fp);
	}	
}

static void test_load_output()
{
	auto layer49 = load_float_text("./160x120_float_result/_model.15_cv3.0_cv3.0.4_Conv_output_0.txt");
	auto layer54 = load_float_text("./160x120_float_result/_model.15_cv2.0_cv2.0.4_Conv_output_0.txt");
	auto layer59 = load_float_text("./160x120_float_result/_model.15_cv3.1_cv3.1.4_Conv_output_0.txt");
	auto layer64 = load_float_text("./160x120_float_result/_model.15_cv2.1_cv2.1.4_Conv_output_0.txt");

	std::map<std::string, stTextInfo> infomap;
	set_info_map(infomap, layer49, 49, 20 * 15);
	set_info_map(infomap, layer54, 54, 20 * 15);
	set_info_map(infomap, layer59, 59, 40 * 30);
	set_info_map(infomap, layer64, 64, 40 * 30);

	auto split0 = load_float_text("./160x120_float_result/_model.15_Split_output_0.txt");
	auto split1 = load_float_text("./160x120_float_result/_model.15_Split_output_1.txt");

	check_info_map(infomap, split0, "split0.txt");
	check_info_map(infomap, split1, "split1.txt");
}

static void find_off(const char* filename, int off)
{
	FILE* fp = fopen(filename, "rt");
	if (fp) {
		float fv;
		int layer, c, off_v;
		while (true) {
			if (4 != fscanf(fp, "%g, layer:%2d c:%2d, off:%4d\n", &fv, &layer, &c, &off_v))
				break;

			if (off_v == off) {
				printf("%f,\n", fv);
			}
		}
		
		fclose(fp);
	}
}

static void find_off_sigmoid(const char* filename, int off)
{
	FILE* fp = fopen(filename, "rt");
	if (fp) {
		float fv;
		int layer, c, off_v;
		while (true) {
			if (4 != fscanf(fp, "%g, layer:%2d c:%2d, off:%4d\n", &fv, &layer, &c, &off_v))
				break;

			if (off_v == off) {
				printf("%f,\n", 1.0f / (1.0f + expf(-fv)));
			}
		}
		printf("---------\n");
		fclose(fp);
	}
}

#include <S16Network/u31/u31_inst.h>
static void softmax(const float* input, float* output, int count)
{
	float sum = 0.0f;
	int i;
	for (i = 0; i < count; i++) {
		output[i] = expf(input[i]);
		sum += output[i];
	}
	float inv_sum = 1.0f / sum;
	for (i = 0; i < count; i++) {
		output[i] *= inv_sum;
	}

	short sdata[10], sum_s = 0;
	for (i = 0; i < count; i++) {
		sdata[i] = u31_exp_m22p10((short)(input[i] * 1024.0f));
		sum_s += sdata[i];
	}
}


static void test_decode()
{
	const float input[40] = {
		4.807097f,	6.469995f,	1.021926f,	-2.220210f,	
		-7.406883f,	-9.106315f,	-9.614369f,	-10.379479f,
		-11.131435f,-12.634628f,12.949838f, 14.390442f,
		9.896819f,	2.744365f,	-1.965793f,	-0.930813f,
		-0.497497f,	-2.951427f,	-9.082280f,	-14.523865f,
		6.868611f,	12.079169f,	11.732760f,	5.919506f,
		2.244221f,	-1.999379f,	-4.641047f,	-6.144914f,
		-7.595733f,	-8.477272f,	3.367610f,	9.697206f,
		13.713607f,	13.271386f,	8.752867f,	1.073669f,
		-6.146559f,	-10.182821f,-12.103513f,-11.359703f,
	};
	const float result[4] = { 47.133347f, 53.127319f, 9.027611f, 12.808559f, };
	float data[40], out[4];
	softmax(input, data, 10);
	softmax(input + 10, data + 10, 10);
	softmax(input + 20, data + 20, 10);
	softmax(input + 30, data + 30, 10);
	int scale = 4;

	int i, j;
	for (i = 0; i < 4; i++) {
		out[i] = 0.0f;
		for (j = 0; j < 10; j++) {
			out[i] += data[i * 10 + j] * j;
		}
	}

	out[0] = 11.5 - out[0];  // w
	out[1] = 12.5 - out[1];  // h
	out[2] = 11.5 + out[2];
	out[3] = 12.5 + out[3];

	float w = (out[2] - out[0]) * scale;
	float h = (out[3] - out[1]) * scale;
	float x = (out[0] + out[2]) / 2 * scale;
	float y = (out[1] + out[3]) / 2 * scale;
}

static void test_softmax()
{
	const float input[] = {
		4.807097f,	6.469995f,	1.021926f,	-2.220210f,
		-7.406883f,	-9.106315f,	-9.614369f,	-10.379479f,
		-11.131435f,-12.634628f
	};
	float out0[10], out1[10];
	int sinput[10], sout0[10], sout1[10];
	float sum = 0.0f;
	int i;
	for (i = 0; i < 10; i++) {
		out0[i] = expf(input[i]);
		sum += out0[i];
	}
	float inv_sum = 1.0f / sum;
	for (i = 0; i < 10; i++) {
		out1[i] = out0[i] * inv_sum;
	}

	int sum_s = 0;
	for (i = 0; i < 10; i++) {
		sinput[i] = (short)(input[i] * 1024.0f);
	}
	for (i = 0; i < 10; i++) {
		sout0[i] = u31_exp_m22p10(sinput[i]);
		sum_s += sout0[i];
	}
	float inv_sum_s = 1024.0f / sum_s;
	for (i = 0; i < 10; i++) {
		sout1[i] = sout0[i] * inv_sum_s;
	}
}

static int inv_sigmoid_int(float score)
{
	float v = (1.0f - score) / score;
	return (int)(-(log(v) * 1024.0f));
}

static void load_bin(const char* filename, cv::Mat& img)
{
	FILE* fp = fopen(filename, "rb");
	fread(img.data, img.cols * img.channels(), img.rows, fp);
	fclose(fp);
}

static void test_320x224()
{
	S16Network* network = new S16Network;
	network->loadNetwork("../data/320x224network.csv", 0);
	network->loadWeight("../data/320x224weight.txt");
	cv::Mat img(224, 320, CV_8UC3);
	load_bin("320x224x3.bin", img);

	short* input_data = get_input_data(img);
	network->forward(input_data);	
	int n = decode_rects(
		network->getLayerOutput(98),
		network->getLayerOutput(100),
		network->getLayerOutput(106),

		network->getLayerOutput(113),
		network->getLayerOutput(115),
		network->getLayerOutput(121)
	);

	for (auto layer : network->layers()) {
		std::string fname = "320x224/layer" + std::to_string(layer->idx) + "_whc.txt";		
		dump_data_whc(fname.c_str(), &layer->output);

		fname = "320x224/layer" + std::to_string(layer->idx) + "_cwh.txt";
		dump_data_whc(fname.c_str(), &layer->output);
	}



	free(input_data);
	delete network;
}

int main(int argc, char** argv)
{
	//test_320x224();
	int v = inv_sigmoid_int(0.22);
	//test_load_output();
	//test_softmax();
	//test_decode();
	//
	//find_off_sigmoid("split1.txt", 492);
	//find_off("split0.txt", 492);


	test_160x120();
	//image2bin(IMAGE_PATH IMAGE_NAME, 320, 224, "320x224x3-a.bin");
	//test_img();
	//image2bin("../data/test1.png", "../data/test1.bin");

	std::vector<stResultContent> content;
	std::vector<stResultRects> u31_res;
	S16Network* network = new S16Network;
	std::string img_dir_path = "";
	int w, h, fmt = -1, c = 0;
	int ok = 0;
	do
	{
		if (argc != 7) break;
		if (!network->loadNetwork(argv[1], 0)) {
			printf("加载网络结构文件(csv)失败!");
			break;
		}

		if (!network->loadWeight(argv[2])) {
			printf("加载网络权重文件(txt)失败!");
			break;
		}
		img_dir_path = argv[3];
		if (img_dir_path.back() != '/')
			img_dir_path.push_back('/');

		if (sscanf(argv[4], "%dx%d", &w, &h) != 2) {
			printf("错误的分辨率:%s", argv[4]);
			break;
		}

		if (strcmp(argv[5], "y") == 0) { 
			fmt = PIXEL_FORMAT_Y;
			c = 1; 
		}
		else if (strcmp(argv[5], "rgb") == 0) { 
			fmt = PIXEL_FORMAT_RGB;
			c = 3; 
		}
		else if (strcmp(argv[5], "bgr") == 0) {
			fmt = PIXEL_FORMAT_BGR;
			c = 3; 
		}
		else {
			printf("错误的像素格式:%s", argv[4]);
			break;
		}

		if (w != network->inputW()
			|| h != network->inputH()
			|| c != network->inputC()) 
		{
			printf("图像格式与网络模型不匹配!\n");
			break;
		}

		if (!decode_result_file(argv[6], u31_res)) {
			break;
		}

		// 对比结果
		for (auto& c : u31_res) {
			std::string filename = img_dir_path + c.name;
			detect_image(network, filename.c_str(), fmt, c);
		}
		ok = 1;
	} while (0);

	if (!ok) {
		usage();
	}
	delete network;
	system("pause");	
	return 0;
}