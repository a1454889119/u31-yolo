#include <filesystem>
#include <opencv2/opencv.hpp>
#define CV_VERSION_ID       CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)

#ifdef _DEBUG
#define cvLIB(name) "opencv_" name CV_VERSION_ID "d"
#else
#define cvLIB(name) "opencv_" name CV_VERSION_ID
#endif
#pragma comment(lib, cvLIB("world"))

#include <UpixNetwork/Layers/UpixBaseLayer.h>
#include <UpixNetwork/UpixNetwork.h>
#include <UpixNetwork/TargetDecoder.h>

#pragma comment(lib, "UpixNetwork.lib")

enum class U31DataType
{
	None,
	Conv_1x1,
	Conv_3x3_Group,	// 3*3分组	
//	Conv_5x5_Group,	// 5*5分组
//	Conv_3x3x3,		// 3输入通道3*3
Mul,
};

struct stPadValue {
	int v;
	int c_start;
	int c_count;
};

struct stU31LayerData {
	int layer_idx = -1;

	std::vector<short> kernels;
	std::vector<int> biases;
	std::vector<stPadValue> pads; // pad values

	bool quantOutput = false;
	int mul_shift;
	float scale;
	float zero_point;

	U31DataType type = U31DataType::None;
	int split = 1;
	int weightSize = 0;
	int outputSize = 0;
	int reflayer = 0;
};


static std::vector<stU31LayerData> makeU31Data(const UpixNetwork& net)
{
	std::vector<stU31LayerData> layerdata;
	for (auto layer : net.layers()) {
		auto info = layer->info();
		stU31LayerData data;
		data.layer_idx = info->idx;
		data.outputSize = layer->output().whc;
		if (info->type == UpixLayerType::Convolution) {

			auto& kernels = layer->kernel().idata;
			data.biases = layer->bias().idata;
			switch (info->ksize) {
			case 3:
				if (info->groups == info->outc || info->inc == 1) {
					data.type = U31DataType::Conv_3x3_Group;
					assert(kernels.size() % 9 == 0);
					for (int i = 0; i < (int)kernels.size(); i += 9) {
						data.kernels.push_back((short)kernels[i + 0]);
						data.kernels.push_back((short)kernels[i + 1]);
						data.kernels.push_back((short)kernels[i + 2]);
						data.kernels.push_back((short)kernels[i + 3]);
						data.kernels.push_back((short)kernels[i + 4]);
						data.kernels.push_back((short)kernels[i + 5]);
						data.kernels.push_back((short)kernels[i + 6]);
						data.kernels.push_back((short)kernels[i + 7]);
						data.kernels.push_back((short)kernels[i + 8]);
						data.kernels.push_back(0);
						data.kernels.push_back(0);
						data.kernels.push_back(0);
						data.kernels.push_back(0);
						data.kernels.push_back(0);
						data.kernels.push_back(0);
						data.kernels.push_back(0);
					}
				}
				else {
					assert(0);
				}

				// set pads
				{
					int c = 0;
					for (auto& from : info->isrcs) {
						data.pads.push_back({ from.pad_value_i, c, from.channel_count });
						c += from.channel_count;
					}
				}
				break;

			case 1:
				data.type = U31DataType::Conv_1x1;
				for (auto v : kernels) {
					data.kernels.push_back((short)v);
				}
				// 最终的输出层不做拆分
				if (info->inc >= 20 && info->refcount) {
					int n = info->outc / info->inc;
					if (n >= 2 && info->inc * n == info->outc) {
						data.split = n;
					}
				}
				break;
			default:
				assert(0);
			}
			data.weightSize = sizeof(short) * (int)data.kernels.size();
			data.weightSize += sizeof(int) * (int)data.biases.size();
			data.scale = info->quantInt.outScale;
			data.zero_point = info->quantInt.outZeroPoint;
			data.mul_shift = info->quantInt.mulShift;
			data.quantOutput = false;
			if (info->quantInt.quantFlag == QuantIntFlag::MulSub) {
				data.quantOutput = true;
			}
			else {
				data.outputSize *= 2;
			}
		}
		else if (info->type == UpixLayerType::Mul) {
			data.scale = info->quantInt.outScale;
			data.zero_point = info->quantInt.outZeroPoint;
			data.mul_shift = info->quantInt.mulShift;
			data.type = U31DataType::Mul;
			data.weightSize = 0;
			data.quantOutput = true;
		}

		// find ref layers
		for (auto other : net.layers()) {
			if (other->info()->idx < info->idx) continue;

			for (auto& from : other->info()->isrcs) {
				if (from.layer == info->idx) {
					if (data.reflayer < other->info()->idx) {
						data.reflayer = other->info()->idx;
					}
				}
			}
		}

		layerdata.push_back(data);
	}

	return layerdata;
}


static void dump_layer_bin(FILE* fp, const stU31LayerData& c)
{
	const short* sptr = c.kernels.data();
	const int* iptr = c.biases.data();
	int kcount = (int)c.kernels.size() / c.split;
	int bcount = (int)c.biases.size() / c.split;
	int i;
	for (i = 0; i < c.split; i++) {
		fwrite(sptr, sizeof(short), kcount, fp);
		fwrite(iptr, sizeof(int), bcount, fp);
		sptr += kcount;
		iptr += bcount;
	}
}

static void dump_bin(const char* filename, const std::vector<stU31LayerData>& layerdata)
{
	FILE* fp = nullptr;
	fopen_s(&fp, filename, "wb");
	if (fp) {
		for (auto& c : layerdata) {
			dump_layer_bin(fp, c);
		}
		fclose(fp);
	}
}


static void dump_quant_params(const char* filename, const std::vector<stU31LayerData>& layerdata)
{
	FILE* fp = nullptr;
	fopen_s(&fp, filename, "wt");
	if (fp) {
		// PAD VALUE放一起
		for (auto& c : layerdata) {
			if (c.pads.size()) {
				if (c.pads.size() == 1) {
					fprintf(fp, "#define LAYER%d_PAD_VALUE 0x%04x\n", c.layer_idx, (c.pads[0].v << 8) | c.pads[0].v);
				}
				else {
					for (auto& p : c.pads) {
						fprintf(fp, "#define LAYER%d_PAD_VALUE_C%d_%d 0x%04x\n", c.layer_idx, p.c_start, p.c_start + p.c_count, (p.v << 8) | p.v);
					}
				}
			}
			fprintf(fp, "#define LAYER%d_MUL_SHIFT %d\n", c.layer_idx, c.mul_shift);
			if (c.quantOutput) {
				fprintf(fp, "#define LAYER%d_QSCALE 0x%08x\n", c.layer_idx, *((int*)&c.scale));
				fprintf(fp, "#define LAYER%d_QZERO_POINT 0x%08x\n", c.layer_idx, *((int*)&c.zero_point));
			}
		}
		fclose(fp);
	}
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

static bool dump_maxmin(const char* networkfilename, const char* weightsfilename, const char* imagepath, const char* outputfilename)
{
	bool res = false;
	UpixNetwork net;
	if (net.loadNetwork(networkfilename)
		&& net.loadWeight(weightsfilename))
	{
		FILE* fp = nullptr;
		fopen_s(&fp, outputfilename, "wt");
		if (fp) {
			auto names = get_img_full_names(imagepath);
			printf("%s目录下有%d个图片\n", imagepath, (int)names.size());

			std::vector<stMaxMin> mms(net.layers().size());
			int total = (int)names.size();
			int i = 0, failed = 0;
			for (auto& name : names) {
				printf("%5d(%3d)/%5d %s\r", i++, failed, total, name.c_str());
				cv::Mat img = cv::imread(name.c_str(), 0);
				if (img.empty()) {
					failed++;
					printf("\r%s文件无效\n", name.c_str());
					continue;
				}
				cv::resize(img, img, cv::Size(320, 160));
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
				printf("layer%d maxv:%g minv:%g\n", i, mm.maxv * 1.02f, mm.minv * 1.02f);
				fprintf(fp, "layer%d maxv:%g minv:%g\n", i, mm.maxv * 1.02f, mm.minv * 1.02f);
			}
			fclose(fp);
			res = true;
		}
		else {
			printf("打开文件%s失败!\n", outputfilename);
		}
	}
	else {
		printf("加载模型文件(%s,%s)失败!\n", networkfilename, weightsfilename);
	}
	return res;
}

static int print_weights_ram_addr(
	FILE* fp,
	const std::vector<stU31LayerData>& layerdata,
	int flash_addr,
	int kernel_region_size,
	int start_idx,
	int end_idx)
{
	static int part_idx = 0;
	int addr = 0;
	int lastIdx = -1;
	int part_size = 0, total_size = 0;
	const std::string baseName = "KERNEL_ADDR";
	for (auto& c : layerdata) {
		if (c.layer_idx < start_idx) continue;
		if (c.layer_idx >= end_idx) break;
		if (c.weightSize) {

			if (addr + c.weightSize > kernel_region_size) {
				lastIdx = -1;
				addr = 0;
				fprintf(fp, "#define WEIGHT_PART_%d_SIZE %d\n", part_idx, part_size);
				fprintf(fp, "#define WEIGHT_PART_%d_FLASH_ADDR 0x%x\n", part_idx, flash_addr);
				part_idx++;
				flash_addr += part_size;
				part_size = 0;
			}

			if (lastIdx < 0) {
				fprintf(fp, "#define LAYER_%d_WEIGHT_RAM_ADDR %s\n", c.layer_idx, baseName.c_str());
			}
			else {
				fprintf(fp, "#define LAYER_%d_WEIGHT_RAM_ADDR (LAYER_%d_WEIGHT_RAM_ADDR + LAYER_%d_WEIGHT_LENGTH)\n", c.layer_idx, lastIdx, lastIdx);
			}
			addr += c.weightSize;
			lastIdx = c.layer_idx;
			part_size += c.weightSize;
			total_size += c.weightSize;
		}
	}

	fprintf(fp, "#define WEIGHT_PART_%d_SIZE %d\n", part_idx, part_size);
	fprintf(fp, "#define WEIGHT_PART_%d_FLASH_ADDR 0x%x\n", part_idx, flash_addr);
	part_idx++;

	return total_size;
}


static void dump_weights_upinc(const char* filename, const std::vector<stU31LayerData>& layerdata, int start_addr)
{
	FILE* fp = nullptr;
	fopen_s(&fp, filename, "wt");
	if (fp) {
		fprintf(fp, "// Weights lengths\n");
		for (auto& c : layerdata) {
			if (c.type == U31DataType::Conv_1x1 || c.type == U31DataType::Conv_3x3_Group) {
				fprintf(fp, "#define LAYER_%d_WEIGHT_LENGTH %d\n", c.layer_idx, c.weightSize);
				fprintf(fp, "#define LAYER_%d_KERNEL_LENGTH %d\n", c.layer_idx, (int)c.kernels.size() * 2);
				fprintf(fp, "#define LAYER_%d_BIAS_LENGTH %d\n", c.layer_idx, (int)c.biases.size() * 4);
				if (c.split > 1) {
					int size = c.weightSize;
					fprintf(fp, "#define LAYER_%d_WEIGHT_PART_LENGTH %d\n", c.layer_idx, size / c.split);
					fprintf(fp, "#define LAYER_%d_KERNEL_PART_LENGTH %d\n", c.layer_idx, (int)c.kernels.size() * 2 / c.split);
					fprintf(fp, "#define LAYER_%d_BIAS_PART_LENGTH %d\n", c.layer_idx, (int)c.biases.size() * 4 / c.split);
				}
			}
			else {
				fprintf(fp, "#define LAYER_%d_WEIGHT_LENGTH %d\n", c.layer_idx, c.weightSize);
			}
		}
		fprintf(fp, "\n");

		fprintf(fp, "// Weights flash addr\n");
		int addr = start_addr;
		for (auto& c : layerdata) {
			fprintf(fp, "#define LAYER_%d_WEIGHT_FLASH_ADDR 0x%x\n", c.layer_idx, addr);
			addr += c.weightSize;
		}
		fprintf(fp, "\n");

		int wsize = print_weights_ram_addr(fp, layerdata, 0x44000, 17 * 1024 - 64, 0, 59);
		print_weights_ram_addr(fp, layerdata, 0x44000 + wsize, 15 * 1024 - 64, 59, 123);

		int classes = (int)layerdata[layerdata.size() - 6].biases.size();
		fprintf(fp, "#define NUM_CLASSES %d\n", classes);
	}
}

#define MODEL_PATH "../../model/"

int main(int argc, char** argv)
{
	//	const char* csv = MODEL_PATH"facepalm_320x160_3types.network.csv";
	//	const char* weights = MODEL_PATH"facepalm_320x160_3types.weights.txt";
	//	const char* maxmin = MODEL_PATH"facepalm_320x160_3types.maxmin.txt";
	//	const char* imgpath = "Z:\\train_data\\facepalm_3types\\images\\val";
	//	bool res = dump_maxmin(csv, weights, imgpath, maxmin);
	//	if (res) {
	//		UpixNetwork net;
	//		if (net.loadNetwork(csv)
	//			&& net.loadWeight(weights)
	//			&& net.loadQuant(maxmin))
	//		{
	//			std::vector<stU31LayerData> layerdata = makeU31Data(net);
	//			dump_quant_params("quant_param.upinc", layerdata);
	//			dump_bin("facepalm_320x160_3types.weights.bin", layerdata);
	//			dump_weights_upinc("weights_flash.upinc", layerdata, 0x44000);
	//		}
	//		else {
	//			printf("加载%s失败!\n", maxmin);
	//		}
	//	}
	//	return 0;
	const char* networkfilename = nullptr;
	const char* weightsfilename = nullptr;
	const char* imagepath = nullptr;
	const char* maxminfilename = nullptr;
	const char* quantfilename = nullptr;
	const char* modelbinfilename = nullptr;
	const char* weightsupinc = nullptr;
	if (argc >= 8) {
		networkfilename = argv[1];
		weightsfilename = argv[2];
		imagepath = argv[3];
		maxminfilename = argv[4];
		quantfilename = argv[5];
		modelbinfilename = argv[6];
		weightsupinc = argv[7];

		bool res = dump_maxmin(networkfilename, weightsfilename, imagepath, maxminfilename);
		if (res) {
			UpixNetwork net;
			if (net.loadNetwork(networkfilename)
				&& net.loadWeight(weightsfilename)
				&& net.loadQuant(maxminfilename))
			{
				std::vector<stU31LayerData> layerdata = makeU31Data(net);
				dump_quant_params(quantfilename, layerdata);
				dump_bin(modelbinfilename, layerdata);
				dump_weights_upinc(weightsupinc, layerdata, 0x44000);
			}
			else {
				printf("加载%s失败!\n", maxminfilename);
			}
		}
	}
	else {
		printf("需要7个参数!\n");
	}
	system("pause");
	return 0;
}