#include <windows.h>
#include <time.h>
#include <thread>
#include <mutex>
#include <filesystem>

#include <opencv2/opencv.hpp>
#define CV_VERSION_ID       CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)

#ifdef _DEBUG
#define cvLIB(name) "opencv_" name CV_VERSION_ID "d"
#else
#define cvLIB(name) "opencv_" name CV_VERSION_ID
#endif
#pragma comment(lib, cvLIB("world"))

//#pragma comment(lib, "Ws2_32.lib")

#include <UpixNetwork/Layers/UpixBaseLayer.h>
#include <UpixNetwork/UpixNetwork.h>
#include <UpixNetwork/TargetDecoder.h>

#pragma comment(lib, "UpixNetwork.lib")



static std::vector<std::string> enumCommNames()
{
	std::vector<std::string> result;
	HKEY hKey;
	char valueName[256];
	char valueData[256];
	DWORD valueNameSize, valueDataSize, type, index = 0;

	if (RegOpenKeyExA(HKEY_LOCAL_MACHINE,
		"HARDWARE\\DEVICEMAP\\SERIALCOMM",
		0,
		KEY_READ,
		&hKey) == ERROR_SUCCESS)
	{
		printf("注册表中的串口:\n");

		while (1) {
			valueNameSize = sizeof(valueName);
			valueDataSize = sizeof(valueData);
			if (RegEnumValueA(hKey, index,
				valueName, &valueNameSize,
				NULL, &type,
				(LPBYTE)valueData, &valueDataSize) != ERROR_SUCCESS)
			{
				break;
			}

			if (atoi(valueData + 3) > 10) {
				result.push_back(std::string("\\\\.\\") + valueData);
			}
			else {
				result.push_back(valueData);
			}

			printf("  %s -> %s\n", valueName, valueData);
			index++;
		}
	}
	else
	{
		printf("无法打开注册表\n");
	}


	RegCloseKey(hKey);
	return result;
}

static HANDLE openCommByName(const char* name)
{
	HANDLE hCom = INVALID_HANDLE_VALUE;
	DCB dcb;
	COMMTIMEOUTS timeouts;

	// 打开串口 (例如 COM1)
	hCom = CreateFileA(
		name,            // 串口名 (要注意 COM10 及以上必须写成 \\\\.\\COM10)
		GENERIC_READ | GENERIC_WRITE,
		0,                        // 独占方式
		NULL,
		OPEN_EXISTING,
		0,                        // 同步方式 (不使用异步)
		NULL);

	if (hCom == INVALID_HANDLE_VALUE) {
		printf("打开串口失败, 错误代码: %lu\n", GetLastError());
		return hCom;
	}

	// 配置串口参数
	SecureZeroMemory(&dcb, sizeof(DCB));
	dcb.DCBlength = sizeof(DCB);

	if (!GetCommState(hCom, &dcb)) {
		printf("获取串口状态失败\n");
		CloseHandle(hCom);
		return INVALID_HANDLE_VALUE;
	}

	dcb.BaudRate = 115200;   // 波特率 115200
	dcb.ByteSize = 8;            // 数据位 8
	dcb.Parity = NOPARITY;     // 无校验
	dcb.StopBits = ONESTOPBIT;   // 1 个停止位

	if (!SetCommState(hCom, &dcb)) {
		printf("配置串口失败\n");
		CloseHandle(hCom);
		return INVALID_HANDLE_VALUE;
	}

	// 配置超时
	timeouts.ReadIntervalTimeout = 50;         // 两字节之间超过50ms算超时
	timeouts.ReadTotalTimeoutMultiplier = 10;  // 每个字节最多等待 10ms
	timeouts.ReadTotalTimeoutConstant = 100;   // 固定超时 100ms
	timeouts.WriteTotalTimeoutMultiplier = 10;
	timeouts.WriteTotalTimeoutConstant = 100;

	if (!SetCommTimeouts(hCom, &timeouts)) {
		printf("配置超时失败\n");
		CloseHandle(hCom);
		return INVALID_HANDLE_VALUE;
	}
	return hCom;
}

struct stTar {
	int score;
	int type;
	int x0;
	int y0;
	int x1;
	int y1;
};

struct stResult {
	int count;
	std::vector<stTar> targets;
};

class ComThread {
public:
	bool quit = false;
	std::mutex mutex;
	std::vector<stResult> rects;
	std::thread* thread = nullptr;	
};

unsigned short htons0(unsigned short v)
{
	return ((v & 0xff) << 8) | (v >> 8);
}

static int face_thres = 64;
static int palm_thres = 64;
char COM_NAME[64] = { 0 };

static void thread_func(void* user)
{
	ComThread* d = (ComThread*)user;
	auto names = enumCommNames();
	if (names.empty()) {
		printf("没有找到串口!\n");
		return;
	}

	const char* com_name = names[0].c_str();
	if (COM_NAME[0] != 0) {
		for (auto& name : names) {
			if (name == std::string("\\\\.\\") + COM_NAME || name == COM_NAME) {
				com_name = name.c_str();
				break;
			}
		}
	}
	printf("|||%s selected|||\n", com_name);
	HANDLE hCom = openCommByName(com_name);
	if (hCom != INVALID_HANDLE_VALUE) {
		DWORD bytesRead;
		unsigned char protocol_data[256];
		
		int nframes = 0; 
		while (!d->quit)
		{
			if (ReadFile(hCom, protocol_data, sizeof(protocol_data), &bytesRead, NULL) && bytesRead > 0) {
				if (protocol_data[0] == 0xfe
					&& protocol_data[1] == 0x0a
					&& protocol_data[bytesRead - 1] == 0x55
					)
				{
					unsigned char* ptr = protocol_data + 2;
					int count = ptr[0];
					// 检查校验和
					int sum = count;
					//if (count * 10 + 5 == bytesRead) 
					{
						int i;
						for (i = 1; i < count * 10 + 1; i++) {
							sum ^= ptr[i];
						}
						if (sum == ptr[i]) {
							stResult res = { count, {} };
							ptr++;
							for (i = 0; i < count; i++) {
								stTar tar;
								tar.x0 = (ptr[1] << 8) | ptr[0];
								tar.y0 = (ptr[3] << 8) | ptr[2];
								tar.x1 = (ptr[5] << 8) | ptr[4];
								tar.y1 = (ptr[7] << 8) | ptr[6];
								tar.score = ptr[8];
								tar.type = ptr[9];
								ptr += 10;

								res.targets.push_back(tar);
								//if (tar.type == 0) {
								//	if (tar.score >= face_thres) {
								//		res.targets.push_back(tar);
								//	}
								//}
								//else {
								//	if (tar.score >= palm_thres) {
								//		res.targets.push_back(tar);
								//	}
								//}
								//printf("u31 %d targes\n", (int)res.targets.size());
							}

							d->mutex.lock();

							d->rects.push_back(res);
							//d->rects.push_back({ protocol.x0, protocol.y0, protocol.x1, protocol.y1 });
							d->mutex.unlock();
						}
						else {
							printf("sum=%d, ptr[i]=%d\n", sum, ptr[i]);
						}
					}
				}
				// 专门给菲亚兰德的
				else if (protocol_data[0] == 0xfe
					&& protocol_data[1] == 0x0b
					&& protocol_data[bytesRead - 1] == 0x55
					&& bytesRead == 5
					)
				{
					printf("亮度(%d)太低!\n", protocol_data[2]);
				}

			}
			else {
				printf("读取失败, 错误代码: %lu\n", GetLastError());
				//break;
			}
		}

		CloseHandle(hCom);
	}
}

static const cv::Scalar COLORS[] = {
	CV_RGB(0x00, 0xff, 0x00),
	CV_RGB(0xff, 0x00, 0x00),
	CV_RGB(0xff, 0xff, 0x00),
	CV_RGB(0x80, 0x00, 0x00),
	CV_RGB(0x00, 0xc0, 0x40),
	CV_RGB(0xc0, 0x40, 0x00),
};

#define MODEL_PATH "../../model/"
#define DATA_PATH "../../data/"
//#define  C2_0_IDX 63
//#define C40_0_IDX 68
//#define  C2_1_IDX 89
//#define C40_1_IDX 94
//#define  C2_2_IDX 117
//#define C40_2_IDX 122

#define  C2_0_IDX 43
#define C40_0_IDX 48
#define  C2_1_IDX 59
#define C40_1_IDX 64

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

static std::vector<Target> getIntTargets(UpixNetwork* net, int num_classes)
{
	std::vector<Target> targets;
	float fthres = face_thres / 255.0f;
	float pthres = palm_thres / 255.0f;

	TargetDecoder dec;
	dec.reset(0.25);
	dec.decode(net->output(C2_0_IDX).iptr, net->output(C40_0_IDX).iptr, net->output(C2_0_IDX).w, net->output(C2_0_IDX).h, num_classes, 8);
	dec.decode(net->output(C2_1_IDX).iptr, net->output(C40_1_IDX).iptr, net->output(C2_1_IDX).w, net->output(C2_1_IDX).h, num_classes, 4);
//	dec.decode(net->output(C2_2_IDX).iptr, net->output(C40_2_IDX).iptr, net->output(C2_2_IDX).w, net->output(C2_2_IDX).h, num_classes, 4);
	for (auto& tar : dec.getTargets()) {
		//if ((tar.type == 0 && tar.score >= fthres)
		//	|| (tar.type == 1 && tar.score >= pthres))
		{
			targets.push_back(tar);
		}
	}
	return targets;
}

static void draw_rects_int(UpixNetwork* net, cv::Mat& img, int num_classes, int color_idx)
{
	float fthres = face_thres / 255.0f;
	float pthres = palm_thres / 255.0f;

	TargetDecoder dec;
	dec.reset(0.25);
	dec.decode(net->output(C2_0_IDX).iptr, net->output(C40_0_IDX).iptr, net->output(C2_0_IDX).w, net->output(C2_0_IDX).h, num_classes, 8);
	dec.decode(net->output(C2_1_IDX).iptr, net->output(C40_1_IDX).iptr, net->output(C2_1_IDX).w, net->output(C2_1_IDX).h, num_classes, 4);
//	dec.decode(net->output(C2_2_IDX).iptr, net->output(C40_2_IDX).iptr, net->output(C2_2_IDX).w, net->output(C2_2_IDX).h, num_classes, 4);
	printf("-----int result\n");
	for (auto& tar : dec.getTargets()) {
		//if ((tar.type == 0 && tar.score >= fthres)
		//	|| (tar.type == 1 && tar.score >= pthres)
		//	|| (tar.type == 2 && tar.score >= pthres)
		//	|| (tar.type == 3 && tar.score >= pthres))
		{
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
}


static void show_image(int idx, bool transpose, int num_classes, const char* csv, const char* weights, const char* maxmin)
{
	cv::Mat input;
	UpixNetwork* netq = nullptr;
	if (csv && weights && maxmin) {
		printf("load model (%s,%s,%s)\n", csv, weights, maxmin);
		netq = new UpixNetwork;
		if (!netq->loadNetwork(csv)
			|| !netq->loadWeight(weights)
			|| !netq->loadQuant(maxmin))
		{
			delete netq;
			netq = nullptr;
		}
	}
	
	cv::VideoCapture vc;
	if (vc.open(idx)) {
		std::vector<stResult> results;
		ComThread thread;
		thread.thread = new std::thread(thread_func, &thread);

		vc.set(CV_CAP_PROP_FRAME_WIDTH, 320);
		vc.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
		vc.set(CV_CAP_PROP_CONVERT_RGB, 0);

		cv::Mat imgShow(160, 320, CV_8UC3);
		cv::Mat img0 = imgShow(cv::Rect(0, 0, 320, 160));

		cv::Mat frame;
		clock_t start = clock();
		int nframes = 0;
		std::filesystem::create_directory("U31IMG");
		while (vc.read(frame)) {
			nframes++;
			printf("FPS:%.3f\r",  (float)nframes*1000 / (float)(clock() - start));

			cv::Mat img;
			
			if (transpose) {
				img = cv::Mat(320, 160, CV_8UC1);
				memcpy(img.data, frame.data, 160 * 320);
				cv::transpose(img, img);
			}
			else {
				img = cv::Mat(160, 320, CV_8UC1);
				memcpy(img.data, frame.data, 160 * 320);
			}
			
			
			//get_img_data_c0_c2(frame.data, img.data, 80 * 320);
			

			thread.mutex.lock();
			results = thread.rects;
			thread.rects.clear();
			thread.mutex.unlock();
			cv::cvtColor(img, img0, cv::COLOR_GRAY2BGR);

			if (results.size()) {
				auto rect = results.back();
				if (rect.targets.size() > 5) {
					int a = 0;
				}

				for (auto& res : results) {
					for (auto& tar : res.targets) {
						cv::rectangle(img0, cv::Rect(tar.x0, tar.y0, tar.x1 - tar.x0, tar.y1 - tar.y0), COLORS[tar.type]);
						printf("U31: type:%d score:%3d, x0:%3d, y0:%3d, x1:%3d, y1:%3d\n",
							tar.type, tar.score, tar.x0, tar.y0, tar.x1, tar.y1);
					}
				}			
			}

			if (netq) {
				cv::Mat img1;
				cv::cvtColor(img, img1, cv::COLOR_GRAY2BGR);

				run_image_int(netq, img);
				draw_rects_int(netq, img1, num_classes, 0);				
				cv::imshow("pc_img", img1);
				
				//cv::transpose(img1, img1);
				//cv::flip(img1, img1, 0);
				//cv::imshow("pc_img", img1);
			}
			cv::imshow("u31_img", imgShow);
			//cv::Mat imgt;
			//cv::transpose(imgShow, imgt);
			//cv::flip(imgt, imgt, 0);
			//cv::imshow("u31_img", imgt);

			int key = cv::waitKey(1);
			if (key == 27) break;
			if (key == ' ') {
				SYSTEMTIME syst;
				GetLocalTime(&syst);

				char filename[64];
				sprintf_s(filename, "./U31IMG/%4d%02d%02d_%02d%02d%02d.png",
					syst.wYear, syst.wMonth, syst.wDay, syst.wHour, syst.wMinute, syst.wSecond);
				cv::imwrite(filename, img);
				printf("%s saved....\n", filename);

				auto targets = getIntTargets(netq, num_classes);
				sprintf_s(filename, "./U31IMG/%4d%02d%02d_%02d%02d%02d.txt",
					syst.wYear, syst.wMonth, syst.wDay, syst.wHour, syst.wMinute, syst.wSecond);
				FILE* fp = nullptr;
				fopen_s(&fp, filename, "w");
				for (auto& tar : targets) {
					fprintf(fp, "%d %g %g %g %g\n", tar.type, tar.x / 320.0f, tar.y / 160.0f, tar.w / 320.0f, tar.h / 160.0f);
				}
			}
		}

		thread.quit = true;
		thread.thread->join();
		delete thread.thread;
	}
}

int main(int argc, char** argv)
{
	int idx = 0;
	int num_classes = 4;
	bool trans = true;
	const char* csv = MODEL_PATH"HeadShoulder_320x160.network.csv";
	const char* weights = MODEL_PATH"HeadShoulder_320x160.weights.txt";
	const char* maxmin = MODEL_PATH"HeadShoulder_320x160.maxmin.txt";

	if (argc > 1) {
		idx = atoi(argv[1]);
	}
	if (argc > 2) {
		float fv = atof(argv[2]);
		face_thres = (int)(255.0f * fv);
	}
	if (argc > 3) {
		float fv = atof(argv[3]);
		palm_thres = (int)(255.0f * fv);
	}
	if (argc > 4) {
		num_classes = atoi(argv[4]);
	}
	if (argc > 5) {
		strcpy_s(COM_NAME, argv[5]);
	}
	if (argc > 6) {
		csv = argv[6];
	}
	if (argc > 7) {
		weights = argv[7];
	}
	if (argc > 8) {
		maxmin = argv[8];
	}


	show_image(idx, trans, num_classes, csv, weights, maxmin);
	cv::destroyAllWindows();
	system("pause");
	return 0;
}