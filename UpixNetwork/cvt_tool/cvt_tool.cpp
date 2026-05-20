#include <assert.h>
#include <vector>
#include <string>
#include <Windows.h>
#include <UpixNetwork/Layers/UpixBaseLayer.h>
#include <UpixNetwork/UpixNetwork.h>

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
						data.kernels.push_back((short)kernels[i+0]);
						data.kernels.push_back((short)kernels[i+1]);
						data.kernels.push_back((short)kernels[i+2]);
						data.kernels.push_back((short)kernels[i+3]);
						data.kernels.push_back((short)kernels[i+4]);
						data.kernels.push_back((short)kernels[i+5]);
						data.kernels.push_back((short)kernels[i+6]);
						data.kernels.push_back((short)kernels[i+7]);
						data.kernels.push_back((short)kernels[i+8]);
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


static void dump_shorts(FILE* fp, const short* sptr, int count)
{
	int i;
	fprintf(fp, ".short ");
	for (i = 0; i < count; i++) {
		fprintf(fp, "0x%04x ", (unsigned short)(*sptr));
		sptr++;

		if ((i + 1) % 8 == 0) {
			fprintf(fp, "\n");
			if (i + 1 != count) {
				fprintf(fp, ".short ");
			}
		}
	}
	fprintf(fp, "\n");
}

static void dump_int(FILE* fp, const int* iptr, int count)
{
	int i;
	fprintf(fp, ".word ");
	for (i = 0; i < count; i++) {
		fprintf(fp, "0x%08x ", (unsigned)(*iptr));
		iptr++;
		if ((i + 1) % 8 == 0) {
			fprintf(fp, "\n");
			if (i + 1 != count) {
				fprintf(fp, ".word ");
			}
		}
	}
	fprintf(fp, "\n");
}

static void dump_layer_upasm(FILE* fp, const stU31LayerData& c)
{
	fprintf(fp, "//---------LAYER%d---------\n", c.layer_idx);
	if (c.kernels.size()) {
		const short* sptr = c.kernels.data();
		const int* iptr = c.biases.data();
		int kcount = (int)c.kernels.size() / c.split;
		int bcount = (int)c.biases.size() / c.split;

		if (c.split == 1) {
			fprintf(fp, "@export layer%d_kernel\n", c.layer_idx);
			printf("@import layer%d_kernel\n", c.layer_idx);
			fprintf(fp, "layer%d_kernel:\n", c.layer_idx);
			dump_shorts(fp, sptr, kcount);
			fprintf(fp, "@export layer%d_bias\n", c.layer_idx);
			fprintf(fp, "layer%d_bias:\n", c.layer_idx);
			dump_int(fp, iptr, bcount);
		}
		else {
			int i;
			for (i = 0; i < c.split; i++) {
				fprintf(fp, "@export layer%d_kernel_p%d\n", c.layer_idx, i);
				printf("@import layer%d_kernel_p%d\n", c.layer_idx, i);
				fprintf(fp, "layer%d_kernel_p%d:\n", c.layer_idx, i);
				dump_shorts(fp, sptr, kcount);
				sptr += kcount;
				fprintf(fp, "@export layer%d_bias_p%d\n", c.layer_idx, i);
				fprintf(fp, "layer%d_bias_p%d:\n", c.layer_idx, i);
				dump_int(fp, iptr, bcount);
				iptr += bcount;
			}
		}
	}
	fprintf(fp, "\n");
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

static void dump_upasm(const char* filename, const std::vector<stU31LayerData>& layerdata)
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

		for (auto& c : layerdata) {
			dump_layer_upasm(fp, c);
		}
		fclose(fp);
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

static void print_weights_size(const std::vector<stU31LayerData>& layerdata)
{
	printf("// Weights lengths\n");
	for (auto& c : layerdata) {
		if (c.type == U31DataType::Conv_1x1 || c.type == U31DataType::Conv_3x3_Group) {
			printf("#define LAYER_%d_WEIGHT_LENGTH %d\n", c.layer_idx, c.weightSize);
			printf("#define LAYER_%d_KERNEL_LENGTH %d\n", c.layer_idx, (int)c.kernels.size() * 2);
			printf("#define LAYER_%d_BIAS_LENGTH %d\n", c.layer_idx, (int)c.biases.size() * 4);
			if (c.split > 1) {
				int size = c.weightSize;
				printf("#define LAYER_%d_WEIGHT_PART_LENGTH %d\n", c.layer_idx, size / c.split);
				printf("#define LAYER_%d_KERNEL_PART_LENGTH %d\n", c.layer_idx, (int)c.kernels.size() * 2 / c.split);
				printf("#define LAYER_%d_BIAS_PART_LENGTH %d\n", c.layer_idx, (int)c.biases.size() * 4 / c.split);
			}
		}
		else {
			printf("#define LAYER_%d_WEIGHT_LENGTH %d\n", c.layer_idx, c.weightSize);
		}
	}
	printf("\n");
}

static void print_weights_addr(int start_addr, const std::vector<stU31LayerData>& layerdata)
{
	printf("// Weights flash addr\n");
	int addr = start_addr;
	for (auto& c : layerdata) {
		printf("#define LAYER_%d_WEIGHT_FLASH_ADDR 0x%x\n", c.layer_idx, addr);
		addr += c.weightSize;
	}
	printf("\n");
}


static void printLayerOutputSize_quant8(const std::vector<UpixBaseLayer*>& layers)
{
	printf("// Layer output lengths\n");
	for (auto& layer : layers) {
		auto c = layer->info();
		int outsize = c->outw * c->outh * c->outc;
		bool towbytes = false;
		switch (c->type) {
		case UpixLayerType::GlobalAvg:
			outsize *= 2;
			towbytes = true;
			break;

		case UpixLayerType::Convolution:
			if (c->quantInt.quantFlag != QuantIntFlag::MulSub) {
				outsize *= 2;
				towbytes = true;
			}
			break;
		}
		printf("#define LAYER_%d_OUTPUT_SIZE %d // %s ",
			c->idx, outsize, towbytes ? "16bit" : "8bit");

		printf("%3dx%3dx%3d->%3dx%3dx%3d ",
			c->inw, c->inh, c->inc, c->outw, c->outh, c->outc);

		switch (c->type) {
		case UpixLayerType::Maxpool:
			printf("maxpool k%ds%dp%d", c->ksize, c->stride, c->pad);
			break;
		case UpixLayerType::Convolution:
			printf("conv k%ds%dp%dg%d", c->ksize, c->stride, c->pad, c->groups);
			switch (c->activate) {
			case ActivateType::Leaky: printf(" leaky"); break;
			case ActivateType::Relu: printf(" relu"); break;
			case ActivateType::Sigmoid: printf(" sigmoid"); break;
			}
			break;

		case UpixLayerType::GlobalAvg:
			printf("global avg");
			break;

		case UpixLayerType::Mul:
			printf("mul");
			break;

		case UpixLayerType::Upsample:
			printf("upsample");
			break;
		}

		printf("\n");
	}
	printf("\n");
}

static int print_weights_ram_addr(
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
				printf("#define WEIGHT_PART_%d_SIZE %d\n", part_idx, part_size);
				printf("#define WEIGHT_PART_%d_FLASH_ADDR 0x%x\n", part_idx, flash_addr);
				part_idx++;
				flash_addr += part_size;
				part_size = 0;
			}

			if (lastIdx < 0) {
				printf("#define LAYER_%d_WEIGHT_RAM_ADDR %s\n", c.layer_idx, baseName.c_str());
			}
			else {
				printf("#define LAYER_%d_WEIGHT_RAM_ADDR (LAYER_%d_WEIGHT_RAM_ADDR + LAYER_%d_WEIGHT_LENGTH)\n", c.layer_idx, lastIdx, lastIdx);
			}
			addr += c.weightSize;
			lastIdx = c.layer_idx;
			part_size += c.weightSize;
			total_size += c.weightSize;
		}
	}

	printf("#define WEIGHT_PART_%d_SIZE %d\n", part_idx, part_size);
	printf("#define WEIGHT_PART_%d_FLASH_ADDR 0x%x\n", part_idx, flash_addr);
	part_idx++;

	return total_size;
}

static int write_weights_ram_addr(
	FILE *fp,
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

		int wsize = write_weights_ram_addr(fp, layerdata, 0x44000, 17 * 1024 - 64, 0, 59);
		write_weights_ram_addr(fp, layerdata, 0x44000 + wsize, 15 * 1024 - 64, 59, 123);

		int classes = (int)layerdata[layerdata.size() - 6].biases.size();
		fprintf(fp, "#define NUM_CLASSES %d\n", classes);
	}
}

#define MODEL_PATH "../model/"
static void cvt_facepalm_320x160()
{
	UpixNetwork net;
	if (!net.loadNetwork(MODEL_PATH"320x160_facepalm.network.csv")) assert(0);
	if (!net.loadWeight(MODEL_PATH"320x160_facepalm.weights.txt")) assert(0);
	if (!net.loadQuant(MODEL_PATH"320x160_maxmin.txt")) assert(0);

	std::vector<stU31LayerData> layerdata = makeU31Data(net);
	dump_upasm("320x160_facepalm_weights.upasm", layerdata);
	dump_bin("320x160_facepalm_weights.bin", layerdata);
	//print_weights_size(layerdata);
	//print_weights_addr(0x44000, layerdata);
	//printLayerOutputSize_quant8(net.layers());
	//int wsize = print_weights_ram_addr(layerdata, 0x44000, 17 * 1024 - 64, 0, 59);
	//print_weights_ram_addr(layerdata, 0x44000 + wsize, 15 * 1024 - 64, 59, 123);
}

static void cvt_320x160()
{
	UpixNetwork net;
	if (!net.loadNetwork(MODEL_PATH"320x160.network.csv")) assert(0);
	if (!net.loadWeight(MODEL_PATH"320x160.weights.txt")) assert(0);
	if (!net.loadQuant(MODEL_PATH"320x160_maxmin.txt")) assert(0);

	std::vector<stU31LayerData> layerdata = makeU31Data(net);
	dump_upasm("320x160_weights.upasm", layerdata);
	dump_bin("320x160_weights.bin", layerdata);
	//print_weights_size(layerdata);
	//print_weights_addr(0x44000, layerdata);
	//printLayerOutputSize_quant8(net.layers());
	int wsize = print_weights_ram_addr(layerdata, 0x44000, 17 * 1024 - 64, 0, 59);
	print_weights_ram_addr(layerdata, 0x44000 + wsize, 15 * 1024 - 64, 59, 123);
}


static void cvt_pool_320x160()
{
	UpixNetwork net;
	if (!net.loadNetwork(MODEL_PATH"Pool_320x160.network.csv")) assert(0);
	if (!net.loadWeight(MODEL_PATH"Pool_320x160.weights.txt")) assert(0);
	if (!net.loadQuant(MODEL_PATH"Pool_320x160.maxmin.txt")) assert(0);

	std::vector<stU31LayerData> layerdata = makeU31Data(net);
	dump_upasm("320x160_pool_weights.upasm", layerdata);
	dump_bin("320x160_pool_weights.bin", layerdata);
	//print_weights_size(layerdata);
	//print_weights_addr(0x44000, layerdata);
	//printLayerOutputSize_quant8(net.layers());
	int wsize = print_weights_ram_addr(layerdata, 0x44000, 17 * 1024 - 64, 0, 125);
	print_weights_ram_addr(layerdata, 0x44000 + wsize, 15 * 1024 - 64, 59, 123);
}

static void cvt_pool_320x160_4classes()
{
	UpixNetwork net;
	if (!net.loadNetwork(MODEL_PATH"Pool_320x160_4classes.network.csv")) assert(0);
	if (!net.loadWeight(MODEL_PATH"Pool_320x160_4classes.weights.txt")) assert(0);
	if (!net.loadQuant(MODEL_PATH"Pool_320x160_4classes.maxmin.txt")) assert(0);

	//int total = 0;
	//for (auto layer : net.layers()) {
	//	total += layer->input().whc + layer->output().whc;
	//}
	//printf("total:%d\n", total);

	std::vector<stU31LayerData> layerdata = makeU31Data(net);
	dump_upasm("Pool_320x160_4classes.upasm", layerdata);
	dump_bin("Pool_320x160_4classes.bin", layerdata);
	dump_weights_upinc("Pool_320x160_4classes_weights_flash.upinc", layerdata, 0x44000);
	//print_weights_size(layerdata);
	//print_weights_addr(0x44000, layerdata);
	////printLayerOutputSize_quant8(net.layers());
	//int wsize = print_weights_ram_addr(layerdata, 0x44000, 17 * 1024 - 64, 0, 122);
	//print_weights_ram_addr(layerdata, 0x44000 + wsize, 15 * 1024 - 64, 59, 122);
}

static void cvt_HeadShoulder_160x128()
{
	UpixNetwork net;
	if (!net.loadNetwork(MODEL_PATH"HeadShoulder_160x128.network.csv")) assert(0);
	if (!net.loadWeight(MODEL_PATH"HeadShoulder_160x128.weights.txt")) assert(0);
	net.quant16();

	std::vector<stU31LayerData> layerdata = makeU31Data(net);
	dump_upasm("HeadShoulder_160x128_weights.upasm", layerdata);
	dump_bin("HeadShoulder_160x128_weights.bin", layerdata);
	//print_weights_size(layerdata);
	//print_weights_addr(0x44000, layerdata);
	//printLayerOutputSize_quant8(net.layers());
	//int wsize = print_weights_ram_addr(layerdata, 0x44000, 17 * 1024 - 64, 0, 59);
	//print_weights_ram_addr(layerdata, 0x44000 + wsize, 15 * 1024 - 64, 59, 123);
}

static void cvt_HeadShoulder_320x160()
{
	UpixNetwork net;
	if (!net.loadNetwork(MODEL_PATH"HeadShoulder_320x160.network.csv")) assert(0);
	if (!net.loadWeight(MODEL_PATH"HeadShoulder_320x160.weights.txt")) assert(0);
	if (!net.loadQuant(MODEL_PATH"HeadShoulder_320x160.maxmin.txt")) assert(0);

	std::vector<stU31LayerData> layerdata = makeU31Data(net);
	dump_upasm("HeadShoulder_320x160_weights.upasm", layerdata);
	dump_bin("HeadShoulder_320x160_weights.bin", layerdata);
	//print_weights_size(layerdata);
	//print_weights_addr(0x44000, layerdata);
	//printLayerOutputSize_quant8(net.layers());
	int wsize0 = print_weights_ram_addr(layerdata, 0x44000, 8 * 1024 - 64, 0, 12);
	int wsize1 = print_weights_ram_addr(layerdata, 0x44000 + wsize0, 14 * 1024 - 64, 12, 33);
	print_weights_ram_addr(layerdata, 0x44000 + wsize0 + wsize1, 12 * 1024 - 64, 33, (int)layerdata.size() + 1);
}


static void cvt_pool_320x160_quant_inc()
{
	UpixNetwork net;
	if (!net.loadNetwork(MODEL_PATH"Pool_320x160.network.csv")) assert(0);
	if (!net.loadWeight(MODEL_PATH"Pool_320x160.weights.txt")) assert(0);
	if (!net.loadQuant(MODEL_PATH"Pool_320x160.maxmin.txt")) assert(0);

	std::vector<stU31LayerData> layerdata = makeU31Data(net);
	dump_quant_params("320x160_pool_weights.upinc", layerdata);
	dump_bin("320x160_pool_weights.bin", layerdata);
}

int main(int argc, char** argv)
{
	cvt_HeadShoulder_320x160();
	system("pause");

	return 0;
}