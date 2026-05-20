#include <filesystem>
#include <opencv2/opencv.hpp>
#include <USB2UARTSPIIICDLL.h>

#include <S16Network/S16Network.h>
#include <S16Network/layers/S16BaseLayer.h>
#include <S16Network/decode_rects.h>


#define CV_VERSION_ID       CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)

#ifdef _DEBUG
#define cvLIB(name) "opencv_" name CV_VERSION_ID "d"
#else
#define cvLIB(name) "opencv_" name CV_VERSION_ID
#endif
#pragma comment(lib, cvLIB("world"))
#pragma comment(lib, "USB2UARTSPIIICDLL")
#pragma comment(lib, "S16Network")

#pragma warning(disable:4996)


#ifndef max
#define max(a, b) ((a) > (b) ? (a) : (b))
#endif // !max

#ifndef min
#define min(a, b) ((a) < (b) ? (a) : (b))
#endif // !max

#define SPI_PACKET_SIZE 1024
static void IO0Set(int state)
{
	IOSetAndRead(0, 1, state, 0);
}

static int spi_recv(unsigned char* ptr, int len)
{
	int remain = len;
	while (remain > 0)
	{
		int recv = min(remain, SPI_PACKET_SIZE);
		recv = SPISlaveRcvData(ptr, recv, 0);
		if (recv < 0) {
			printf("SPISlaveRcvData error:%d\n", recv);
			assert(0);
		}

		ptr += recv;
		remain -= recv;
	}
	return len;
}

static int spi_send_recv(unsigned char* ptr, int len)
{
	int remain = len;
	while (remain > 0)
	{
		int preload = min(remain, SPI_PACKET_SIZE);
		IO0Set(1);
		int r = SPISlavePreloadData(ptr, preload, 0);
		IO0Set(0);
		if (r != 0) {
			printf("SPISlavePreloadData error:%d", r);
		}
		spi_recv(ptr, preload);

		ptr += preload;
		remain -= preload;
	}
	return len;
}

#define CSV_NAME "../data/320x224network.csv"
#define WEIGHT_NAME "../data/320x224weight.txt"
#define IMAGE_PATH "\\\\192.168.1.173\\优象资料库\\研发部\\研发部\\01-测试数据wdz\\06-YOLO数据集\\Png\\"

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



static const cv::Scalar COLORS[] = {
	CV_RGB(0x00, 0xff, 0x00),
	CV_RGB(0xff, 0x00, 0x00),
	CV_RGB(0x00, 0x00, 0xff),
	CV_RGB(0x80, 0x80, 0x00),
	CV_RGB(0x00, 0x80, 0x80),
	CV_RGB(0x80, 0x00, 0x80),
	CV_RGB(0xff, 0x00, 0x80),
	CV_RGB(0x40, 0xff, 0x40),
};

static void draw_rects(cv::Mat& img, const std::vector<TargetRect>& rects)
{
	for (auto& r : rects) {
		cv::rectangle(img, cv::Rect(r.x >> 10, r.y >> 10, r.w >> 10, r.h >> 10), COLORS[r.type]);
	}
}

static void get_input_data(const cv::Mat& img, short* data)
{
	int wh = img.rows * img.cols;
	const unsigned char* src_ptr = img.data;
	short* input_data = data;

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

#define PIXEL_FORMAT_Y 0
#define PIXEL_FORMAT_RGB 1
#define PIXEL_FORMAT_BGR 2

static std::vector<TargetRect> detect_image(S16Network* network, const short* input_data, int fmt)
{
	std::vector<TargetRect> result;
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
	return result;
}


static void test_images(int start_idx, const char* imgpath)
{
	auto names = get_img_full_names(imgpath);
	if (start_idx < 0 || start_idx >= (int)names.size()) {
		return;
	}

	S16Network* network = new S16Network;
	network->loadNetwork(CSV_NAME, 0);
	network->loadWeight(WEIGHT_NAME);
	int w = network->inputW();
	int h = network->inputH();
	int c = network->inputC();
	short* input_data = (short*)malloc(sizeof(short) * w * h * c);
	cv::Mat img, dstimg;
	int idx;
	for (idx = start_idx; idx < names.size(); idx++) {
		auto name = names[idx];
		img = cv::imread(name);
		if (!img.empty()) {
			
			if (img.rows != h || img.cols != w) {
				float wratio = (float)img.cols / w;
				float hratio = (float)img.rows / h;

				if (wratio <= hratio) {
					int tmph = (int)(img.rows / wratio);
					cv::resize(img, img, cv::Size(w, tmph));
					img(cv::Rect(0, tmph / 2, w, h)).copyTo(dstimg);
				}
				else {
					int tmpw = (int)(img.cols / hratio);
					cv::resize(img, img, cv::Size(tmpw, h));
					img(cv::Rect(tmpw / 2, 0, w, h)).copyTo(dstimg);
				}
			}
			get_input_data(dstimg, input_data);
			auto rects = detect_image(network, input_data, PIXEL_FORMAT_BGR);			
			draw_rects(dstimg, rects);

			cv::imshow("yolo", img);
			int key = cv::waitKey(1);
			if (key == 27) break;
		}
	}
	cv::destroyAllWindows();
	free(input_data);
	delete network;
}

#define YOLO_U31_RESULT_SIZE 1024


static std::vector<stTargetRect> decode_u31_rects(const unsigned char* data)
{
	std::vector<stTargetRect> rects;
	const int* pdata = (const int*)data;
	int n = (pdata[0] - 0x19000) / 32;
	if (n < 0 || n * 32 + 4 > YOLO_U31_RESULT_SIZE) {
		assert(0);
	}

	pdata++;
	for (int i = 0; i < n; i++, pdata += 8) {
		rects.push_back({ pdata[0], pdata[1], pdata[4], pdata[5], pdata[6], pdata[7] });
	}
	return rects;
}

static bool same_rects(const std::vector<stTargetRect>& r0, const std::vector<stTargetRect>& r1)
{
	if (r0.size() == r1.size()
		&& memcmp(r0.data(), r1.data(), sizeof(stTargetRect) * r0.size()) == 0)
	{
		return true;
	}
	return false;
}

static void print_diff(const std::vector<stTargetRect>& rects0, const std::vector<stTargetRect>& rects1)
{
	if (rects0.size() != rects1.size()) {
		printf("目标数量不同!!!\n"); 
		return;
	}
	else {
		printf("共%d个目标\n", (int)rects0.size());
		int i;
		for (i = 0; i < (int)rects0.size(); i++) {
			auto& r0 = rects0[i];
			auto& r1 = rects1[i];
			if (memcmp(&r0, &r1, sizeof(r0)) != 0) {
				
				printf("  目标%d ", i);
				if (r0.type != r1.type) {
					printf("类型不同!");
				}
				else {
					if (r0.score != r1.score) {
						printf("得分差%d ", (r0.score - r1.score));
					}
					if (r0.x != r1.x) {
						printf("x差%d ", (r0.x - r1.x));
					}
					if (r0.y != r1.y) {
						printf("y差%d ", (r0.y - r1.y));
					}
					if (r0.w != r1.w) {
						printf("w差%d ", (r0.w - r1.w));
					}
					if (r0.h != r1.h) {
						printf("h差%d ", (r0.h - r1.h));
					}
				}
				printf("\n");
			}
		}
	}
}

static void compare_u31(int start_idx, const char* imgpath)
{
	auto names = get_img_full_names(imgpath);
	if (start_idx < 0 || start_idx >= (int)names.size()) {
		return;
	}

	int r = OpenUsb(0);
	if (r != 0) {
		printf("打开USB设备失败 %d\n", r);
		return;
	}

	unsigned char UID[12]; //UID
	unsigned int isHsUSB; //USB 速度类型
	unsigned int firmwareVersion; //固件版本
	GetBoardInformation(UID, &isHsUSB, &firmwareVersion, 0);

	r = ConfigSPIParamSlave(SPI_MSB, 0, 0);
	if (r != 0) {
		printf("配置SPI接口失败 %d\n", r);
		CloseUsb(0);
		return;
	}

	S16Network* network = new S16Network;
	network->loadNetwork(CSV_NAME, 0);
	network->loadWeight(WEIGHT_NAME);

	int w = network->inputW();
	int h = network->inputH();
	int c = network->inputC();
	unsigned char recv_data[YOLO_U31_RESULT_SIZE];

	short* input_data = (short*)malloc(sizeof(short) * w * h * c);
	cv::Mat img, img_show(h, w*2, CV_8UC3), img_send;
	cv::Mat img_pc = img_show(cv::Rect(0, 0, w, h));
	cv::Mat img_u31 = img_show(cv::Rect(w, 0, w, h));
	

	int idx;
	for (idx = start_idx; idx < names.size(); idx++) {
		auto name = names[idx];
		img = cv::imread(name);
		if (!img.empty()) {
			std::filesystem::path p(name);

			IO0Set(1);
			if (img.rows != h || img.cols != w) {
				cv::resize(img, img, cv::Size(w, h));
			}
			img.copyTo(img_pc);
			img.copyTo(img_u31);
			img_send = img.clone();
			IO0Set(0);

			// SPI发送帧数据
			spi_send_recv(img_send.data, w * h * c);

			// 执行算法
			get_input_data(img, input_data);
			auto pc_rects = detect_image(network, input_data, PIXEL_FORMAT_BGR);

			// SPI接收u31结果
			memset(recv_data, 0, YOLO_U31_RESULT_SIZE);
			spi_send_recv(recv_data, YOLO_U31_RESULT_SIZE);

			// 解析U31结果
			auto u31_rects = decode_u31_rects(recv_data);

			// 比较结果
			bool same = same_rects(pc_rects, u31_rects);
			if (same) {
				printf("                                                             \r");
				printf("frame(%d) %s 完全相同\r", idx, p.filename().string().c_str());
			}
			else {
				printf("frame(%d) %s 不同!\n", idx, p.filename().string().c_str());
				print_diff(pc_rects, u31_rects);
			}

			draw_rects(img_pc, pc_rects);
			draw_rects(img_u31, u31_rects);
			cv::imshow("yolo", img_show);
			int key = cv::waitKey(same ? 1 : 0);
			if (key == 27) break;
		}
	}
	printf("\n");
	cv::destroyAllWindows();
	free(input_data);
	delete network;
	CloseUsb(0);
}

static void compare_u31_dump(int start_idx, const char* input_name, const char* output_name, const char* imgpath)
{
	auto names = get_img_full_names(imgpath);
	if (start_idx < 0 || start_idx >= (int)names.size()) {
		return;
	}

	int r = OpenUsb(0);
	if (r != 0) {
		printf("打开USB设备失败 %d\n", r);
		return;
	}

	unsigned char UID[12]; //UID
	unsigned int isHsUSB; //USB 速度类型
	unsigned int firmwareVersion; //固件版本
	GetBoardInformation(UID, &isHsUSB, &firmwareVersion, 0);

	r = ConfigSPIParamSlave(SPI_MSB, 0, 0);
	if (r != 0) {
		printf("配置SPI接口失败 %d\n", r);
		CloseUsb(0);
		return;
	}
	FILE* fp_in = fopen(input_name, "wb");
	FILE* fp_out = fopen(output_name, "wb");

	S16Network* network = new S16Network;
	network->loadNetwork(CSV_NAME, 0);
	network->loadWeight(WEIGHT_NAME);

	int w = network->inputW();
	int h = network->inputH();
	int c = network->inputC();
	unsigned char recv_data[YOLO_U31_RESULT_SIZE];

	short* input_data = (short*)malloc(sizeof(short) * w * h * c);
	cv::Mat img, img_show(h, w * 2, CV_8UC3), img_send;
	cv::Mat img_pc = img_show(cv::Rect(0, 0, w, h));
	cv::Mat img_u31 = img_show(cv::Rect(w, 0, w, h));

	int idx;
	for (idx = start_idx; idx < names.size(); idx++) {
		auto name = names[idx];
		img = cv::imread(name);
		if (!img.empty()) {
			std::filesystem::path p(name);

			IO0Set(1);
			if (img.rows != h || img.cols != w) {
				cv::resize(img, img, cv::Size(w, h));
			}
			img.copyTo(img_pc);
			img.copyTo(img_u31);
			img_send = img.clone();
			IO0Set(0);

			// SPI发送帧数据
			spi_send_recv(img_send.data, w * h * c);

			// 执行算法
			get_input_data(img, input_data);
			auto pc_rects = detect_image(network, input_data, PIXEL_FORMAT_BGR);

			// SPI接收u31结果
			memset(recv_data, 0, YOLO_U31_RESULT_SIZE);
			spi_send_recv(recv_data, YOLO_U31_RESULT_SIZE);

			// 解析U31结果
			auto u31_rects = decode_u31_rects(recv_data);

			// 比较结果
			bool same = same_rects(pc_rects, u31_rects);
			if (same) {
				printf("                                                             \r");
				printf("frame(%d) %s 完全相同\r", idx, p.filename().string().c_str());
			}
			else {
				printf("frame(%d) %s 不同!\n", idx, p.filename().string().c_str());
				print_diff(pc_rects, u31_rects);
			}

			draw_rects(img_pc, pc_rects);
			draw_rects(img_u31, u31_rects);
			cv::imshow("yolo", img_show);

			fwrite(img.data, w, h*c, fp_in);
			fwrite(recv_data, 1, YOLO_U31_RESULT_SIZE, fp_out);

			int key = cv::waitKey(same ? 1 : 0);
			if (key == 27) break;
		}
	}
	fclose(fp_in);
	fclose(fp_out);
	printf("\n");
	cv::destroyAllWindows();
	free(input_data);
	delete network;
	CloseUsb(0);
}


static void test_1_image(const char* filename)
{
	S16Network* network = new S16Network;
	network->loadNetwork(CSV_NAME, 0);
	network->loadWeight(WEIGHT_NAME);

	int w = network->inputW();
	int h = network->inputH();
	int c = network->inputC();
	short* input_data = (short*)malloc(sizeof(short) * w * h * c);
	cv::Mat img = cv::imread(filename);
	if (!img.empty()) {
		if (img.rows != h || img.cols != w) {
			cv::resize(img, img, cv::Size(w, h));
		}
		get_input_data(img, input_data);
		auto rects = detect_image(network, input_data, PIXEL_FORMAT_BGR);
		draw_rects(img, rects);

		cv::imshow("yolo", img);
		cv::waitKey(0);
	}
	cv::destroyAllWindows();
	free(input_data);
	delete network;
}

#define OUTPUT_PATH "D:\\data\\u31仿真激励\\yolo-320x224\\"

static void print_rects(const std::vector<TargetRect>& rects) {
	for (auto& r : rects) {
		printf("%d %d %d %d %d %d\n", r.x, r.y, r.w, r.h, r.score, r.type);
	}	
}

static void fprint_rects(FILE *fp, const std::vector<TargetRect>& rects) {
	for (auto& r : rects) {
		fprintf(fp, "%d %d %d %d %d %d\n", r.x, r.y, r.w, r.h, r.score, r.type);
	}
}

#include <windows.h>

static void run_cmos_spi()
{
	int r = OpenUsb(0);
	if (r != 0) {
		printf("打开USB设备失败 %d\n", r);
		return;
	}

	unsigned char UID[12]; //UID
	unsigned int isHsUSB; //USB 速度类型
	unsigned int firmwareVersion; //固件版本
	GetBoardInformation(UID, &isHsUSB, &firmwareVersion, 0);

	r = ConfigSPIParamSlave(SPI_MSB, 0, 0);
	if (r != 0) {
		printf("配置SPI接口失败 %d\n", r);
		CloseUsb(0);
		return;
	}

	struct _SYSTEMTIME t;
	///struct tm t;
	GetLocalTime(&t);
	char logname[1024];
	sprintf(logname, "%4d_%2d_%2d__%2d_%2d_%2d.txt",
		t.wYear, t.wMonth, t.wDay,
		t.wHour, t.wMinute, t.wSecond);
	FILE* fp = fopen(logname, "wt");



	S16Network* network = new S16Network;
	network->loadNetwork(CSV_NAME, 0);
	network->loadWeight(WEIGHT_NAME);

	int w = network->inputW();
	int h = network->inputH();
	int c = network->inputC();
	unsigned char recv_data[YOLO_U31_RESULT_SIZE];

	short* input_data = (short*)malloc(sizeof(short) * w * h * c);
	cv::Mat img_show(h, w * 2, CV_8UC3), img_recv(h, w, CV_8UC3);
	
	cv::Mat img_pc = img_show(cv::Rect(0, 0, w, h));
	cv::Mat img_u31 = img_show(cv::Rect(w, 0, w, h));
	int idx = 0;
	while (true)
	{
		IO0Set(1);
		_sleep(100);
		IO0Set(0);

		// SPI接收帧数据
		spi_send_recv(img_recv.data, w * h * c);

		img_recv.copyTo(img_pc);
		img_recv.copyTo(img_u31);

		cv::cvtColor(img_show, img_show, CV_BGR2RGB);
		// SPI接收u31结果
		memset(recv_data, 0, YOLO_U31_RESULT_SIZE);
		

		// 执行算法
		get_input_data(img_recv, input_data);
		auto pc_rects = detect_image(network, input_data, PIXEL_FORMAT_BGR);

		spi_send_recv(recv_data, YOLO_U31_RESULT_SIZE);
		// 解析U31结果
		auto u31_rects = decode_u31_rects(recv_data);

		fprintf(fp, "----------------\n");
		fprintf(fp, "pc result(%d):\n", (int)pc_rects.size());
		fprint_rects(fp, pc_rects);
		fprintf(fp, "u31 result(%d):\n", (int)u31_rects.size());
		fprint_rects(fp, u31_rects);

		//// 比较结果
		//bool same = same_rects(pc_rects, u31_rects);
		//if (same) {
		//	printf("                                                             \r");
		//	printf("frame(%d) %d个目标 完全相同\r", idx, (int)pc_rects.size());
		//}
		//else {
		//	printf("frame(%d) 不同!\n", idx);
		//	print_diff(pc_rects, u31_rects);
		//}

		draw_rects(img_pc, pc_rects);
		draw_rects(img_u31, u31_rects);
		cv::imshow("yolo", img_show);
		if (idx % 100 == 0) {
			SYSTEMTIME time;
			GetLocalTime(&time);

			char name[128];
			sprintf(name, "%4d-%02d-%02d_%02d-%02d-%02d.png",
				time.wYear, time.wMonth, time.wDay, time.wHour, time.wMinute, time.wSecond);
			cv::imwrite(name, img_show);
		}

		int key = cv::waitKey(1);
		if (key == 27) break;
		idx++;
	}

	fclose(fp);
	//printf("\n");
	cv::destroyAllWindows();
	free(input_data);
	delete network;
	CloseUsb(0);
}

void showdiff()
{
	cv::Mat img1(224, 320, CV_8UC3);
	cv::Mat img2(224, 320, CV_8UC3);
	FILE* fptmp = fopen("temp_320x224x3.bin", "rb");
	fread(img1.data, 320, 224 * 3, fptmp);
	fclose(fptmp);

	fptmp = fopen("320x224x3_psram.bin", "rb");
	fread(img2.data, 320, 224 * 3, fptmp);
	fclose(fptmp);

	cv::Mat diff;
	cv::absdiff(img1, img2, diff);
}

static std::vector<std::filesystem::path> get_img_paths(const char* imgpath)
{
	std::vector<std::filesystem::path> names;
	auto dir = std::filesystem::directory_iterator(imgpath);
	for (auto& fe : dir) {
		auto fp = fe.path();
		if (fe.is_regular_file() && (fp.extension() == ".png" || fp.extension() == ".jpg")) {
			names.push_back(fp);
		}
	}
	return names;
}

static bool load_txt_rects(const char* filename, int imgw, int imgh, int x0, int y0, std::vector<stTargetRect>& res)
{
	res.clear();
	FILE* fp = fopen(filename, "rt");
	if (fp) {
		int type;
		float cx, cy, w, h;
		while (true)
		{
			if (fscanf(fp, "%d %g %g %g %g\n", &type, &cx, &cy, &w, &h) != 5)
				break;

			stTargetRect target;
			switch (type) {
			case 0: case 1: case 2: case 3:
				target.type = 0;
				break;

			case 4: 
				target.type = 2;
				break;
			case 5:
				target.type = 1;
				break;
			default:
				break;
			}
			target.score = -1;
			target.x = (int)((cx - w * 0.5) * imgw * 1024.0f);// + x0 * 1024;
			target.y = (int)((cy - h * 0.5) * imgh * 1024.0f);// + y0 * 1024;
			target.w = (int)(w * imgw * 1024.0f);
			target.h = (int)(h * imgh * 1024.0f);
			res.push_back(target);
		}
		fclose(fp);
		return true;
	}
	return false;
}

struct stTypeStatisticInfo {
	int lost = 0;
	int error = 0;
	int good = 0;
};

static void compare_rects(
	std::vector<stTargetRect>& gt, 
	std::vector<stTargetRect>& res,
	stTypeStatisticInfo* infos)
{
	int x0, x1, y0, y1, w, h;
	for (auto& tar0 : gt) {
		for (size_t i = 0; i < res.size(); i++) {
			auto& tar1 = res[i];
			if (tar1.score == 0)continue;
			if (tar1.type != tar0.type) continue;
			if (tar0.w > tar1.w * 2 || tar1.w > tar0.w * 2) continue;
			if (tar0.h > tar1.h * 2 || tar1.h > tar0.h * 2) continue;

			x0 = max(tar0.x, tar1.x);
			x1 = min(tar0.x + tar0.w, tar1.x + tar1.w);
			w = x1 - x0;
			if (w <= 0) continue;

			if (w < tar0.w / 2 || w < tar1.w / 2)continue;

			y0 = max(tar0.y, tar1.y);
			y1 = min(tar0.y + tar0.h, tar1.y + tar1.h);
			h = y1 - y0;
			if (h <= 0) continue;
			if (h < tar0.h / 2 || h < tar1.h / 2)continue;

			tar0.score = (int)i;
			tar1.score = 0;
			break;
		}
	}

	for (auto& tar : gt) {
		if (tar.score >= 0) {
			infos[tar.type].good++;
		}
		else {
			infos[tar.type].lost++;
		}
	}

	for (auto& tar : res) {
		if (tar.score != 0) {
			infos[tar.type].error++;
		}
	}
}

static void test_images(const char* imgpath, const char* txtpath)
{
	auto names = get_img_paths(imgpath);
	stTypeStatisticInfo infos[3];
	S16Network* network = new S16Network;
	network->loadNetwork(CSV_NAME, 0);
	network->loadWeight(WEIGHT_NAME);
	int w = network->inputW();
	int h = network->inputH();
	int c = network->inputC();
	short* input_data = (short*)malloc(sizeof(short) * w * h * c);
	cv::Mat img, dstimg(h, w, CV_8UC3);
	int idx, x0, y0;
	std::filesystem::path txtp(txtpath);
	std::vector<stTargetRect> gtres, netres;
	for (idx = 0; idx < names.size(); idx++) {
		auto imgname = names[idx].string();
		auto txtname = (txtp / names[idx].filename()).replace_extension("txt");

		img = cv::imread(imgname);
		if (img.empty()) continue;
		x0 = 0, y0 = 0;
		if (img.rows != h || img.cols != w) {
			float wratio = (float)img.cols / w;
			float hratio = (float)img.rows / h;

			if (wratio <= hratio) {
				int tmph = (int)(img.rows / wratio);
				cv::resize(img, img, cv::Size(w, tmph));
				y0 = (tmph - h) / 2;
				img(cv::Rect(0, (tmph - h) / 2, w, h)).copyTo(dstimg);
			}
			else {
				int tmpw = (int)(img.cols / hratio);
				cv::resize(img, img, cv::Size(tmpw, h));
				x0 = (tmpw - w) / 2;
				img(cv::Rect((tmpw - w) / 2, 0, w, h)).copyTo(dstimg);
			}
		}

		if (!load_txt_rects(txtname.string().c_str(), w, h, x0, y0, gtres)) continue;
		
		get_input_data(dstimg, input_data);
		auto netres = detect_image(network, input_data, PIXEL_FORMAT_BGR);
		
		
		//cv::Mat gtimg = dstimg.clone();
		//cv::Mat resimg = dstimg.clone();
		//
		//draw_rects(gtimg, gtres);
		//draw_rects(resimg, netres);
		printf("idx:%d %s\n", idx, names[idx].filename().string().c_str());
		compare_rects(gtres, netres, infos);
	}
	printf("hand good:%d, lost:%d, error:%d\n", infos[0].good, infos[0].lost, infos[0].error);
	printf("pedestrain good:%d, lost:%d, error:%d\n", infos[1].good, infos[1].lost, infos[1].error);
	printf("face good:%d, lost:%d, error:%d\n", infos[2].good, infos[2].lost, infos[2].error);
	
	free(input_data);
	delete network;
}

#define TEST_IMAGE_PATH "\\\\192.168.1.57\\Ai人脸人行项目文件\\05-人形人脸检测20250714\\val_yx\\val_yx"
#define TEST_TXT_PATH "\\\\192.168.1.57\\Ai人脸人行项目文件\\05-人形人脸检测20250714\\val_yx_txt\\val_yx"
int main(int argc, char** argv)
{
	//test_images(TEST_IMAGE_PATH, TEST_TXT_PATH);
	//compare_u31(0, IMAGE_PATH);
	run_cmos_spi();
	return 0;
	//test_images(0, )
	//showdiff();
	//int a = arith_op_exp2_m16p16(0x435678ab);

//	cv::Mat img(240, 320, CV_8UC3);
//	FILE* fp = fopen("320x240x3--psram.bin", "rb");
//	fread(img.data, 320, 240 * 3, fp);
//	fclose(fp);

//	test_images(61);
 	const char* imgpath = IMAGE_PATH;
	if (argc == 2) {
		imgpath = argv[1];
	}
	run_cmos_spi();
	compare_u31(0, imgpath);
	
//	compare_u31_dump(0, OUTPUT_PATH"input.bin", OUTPUT_PATH"output.bin");
	system("pause");
	return 0;
}	