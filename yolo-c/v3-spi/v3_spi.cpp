#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <opencv2/opencv.hpp>

#include <S16Network/S16Network.h>
#include <S16Network/layers/S16BaseLayer.h>
#include <S16Network/decode_rects_160x120.h>
#include <USB2UARTSPIIICDLL.h>

#define CV_VERSION_ID       CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)

#ifdef _DEBUG
#define cvLIB(name) "opencv_" name CV_VERSION_ID "d"
#else
#define cvLIB(name) "opencv_" name CV_VERSION_ID
#endif
#pragma comment(lib, cvLIB("world"))
#pragma comment(lib, "S16Network")
#pragma comment(lib, "USB2UARTSPIIICDLL")


#pragma warning(disable:4996)

#ifndef max
#define max(a, b) ((a) > (b) ? (a) : (b))
#endif // !max

#ifndef min
#define min(a, b) ((a) < (b) ? (a) : (b))
#endif // !max

#define SPI_PACKET_SIZE 1280
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

class SPI {
public:
	SPI() {}
	~SPI()
	{
		CloseUsb(0);
	}

	bool connect() {
		int r = OpenUsb(0);
		if (r != 0) {
			printf("打开USB设备失败 %d\n", r);
			return false;
		}

		unsigned char UID[12]; //UID
		unsigned int isHsUSB; //USB 速度类型
		unsigned int firmwareVersion; //固件版本
		GetBoardInformation(UID, &isHsUSB, &firmwareVersion, 0);

		r = ConfigSPIParamSlave(SPI_MSB, 0, 0);
		if (r != 0) {
			printf("配置SPI接口失败 %d\n", r);
			return false;
		}
		return true;
	}
};


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

static std::vector<Target2Type> detect_image(S16Network* network, const short* input_data)
{
	std::vector<Target2Type> result;
	network->forward((short*)input_data);
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

	const Target2Type* prect = get_decoded_rects_160x120();
	int i;
	for (i = 0; i < n; i++, prect++) {
		result.push_back(*prect);
	}
	return result;
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

static void draw_rects(cv::Mat& img, const std::vector<Target2Type>& rects)
{
	for (auto& r : rects) {
		cv::Rect cvr;
		cvr.x = (r.x - r.w / 2) >> 10;
		cvr.y = (r.y - r.h / 2) >> 10;
		cvr.width =  r.w >> 10;
		cvr.height = r.h >> 10;

		cv::rectangle(img, cvr, COLORS[r.type]);
	}
}

static std::vector<Target2Type> decode_u31_rects(const unsigned char* data)
{
	std::vector<Target2Type> rects;
	const int* pdata = (const int*)data;
	int n = pdata[0];
	if (n < 0 || n > 32) {
		assert(0);
	}

	pdata += 2;
	for (int i = 0; i < n; i++, pdata += 6) {
		rects.push_back({ pdata[0], pdata[1], pdata[2], pdata[3], pdata[4], pdata[5] });
	}
	return rects;
}

static void print_rects(const std::vector<Target2Type>& rects) {
	for (auto& r : rects) {
		printf("%d %d %d %d %4d %d\n", r.x, r.y, r.w, r.h, r.score, r.type);
	}
}


#define CSV_NAME "../data/160x120.network.csv"
#define WEIGHT_NAME "../data/160x120.weights.txt"

static void test_cmos_spi()
{
	SPI spi;
	if (!spi.connect())
		return;

	S16Network* network = new S16Network;
	network->loadNetwork(CSV_NAME, 1);
	network->loadWeight(WEIGHT_NAME);

	int w = network->inputW();
	int h = network->inputH();
	int c = network->inputC();
	unsigned char recv_data[SPI_PACKET_SIZE];
	short* input_data = (short*)malloc(sizeof(short) * w * h * c);
	assert(input_data);

	cv::Mat img_show(h, w * 2, CV_8UC1);
	cv::Mat img_recv(w, h, CV_8UC1);
	

	cv::Mat img_pc = img_show(cv::Rect(0, 0, w, h));
	cv::Mat img_u31 = img_show(cv::Rect(w, 0, w, h));
	int idx = 0;

	while (true)
	{
		IO0Set(1);
		_sleep(100);
		IO0Set(0);

		img_recv = 0;
		// SPI接收帧数据
		spi_send_recv(img_recv.data, w * h * c);

		cv::Mat img_recv_trans;
		cv::transpose(img_recv, img_recv_trans);

		img_recv_trans.copyTo(img_pc);
		img_recv_trans.copyTo(img_u31);

		//cv::cvtColor(img_show, img_show, CV_BGR2RGB);
		// SPI接收u31结果
		memset(recv_data, 0, SPI_PACKET_SIZE);


		// 执行算法
		get_input_data(img_recv_trans, input_data);
		auto pc_rects = detect_image(network, input_data);

		spi_send_recv(recv_data, SPI_PACKET_SIZE);
		// 解析U31结果
		auto u31_rects = decode_u31_rects(recv_data);

		printf("----------------\n");
		printf("pc result(%d):\n", (int)pc_rects.size());
		print_rects(pc_rects);
		printf("u31 result(%d):\n", (int)u31_rects.size());
		print_rects(u31_rects);

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
		
		int key = cv::waitKey(1);
		if (key == 27) break;
		idx++;
	}


	delete network;
	free(input_data);
	cv::destroyAllWindows();
}


int main(int argc, char** argv)
{
	test_cmos_spi();
	system("pause");
	return 0;
}