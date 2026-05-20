#include <assert.h>
#include <memory.h>
#include <stdio.h>

#include <string>
#include <vector>
#include <map>

#include <S16Network/S16Network.h>
#include <S16Network/layers/S16ConvLayer.h>

#include <util_func/util_func.hpp>

#pragma comment(lib, "S16Network")
#pragma warning(disable:4996)

enum enU31ConvType
{
	Conv_1x1,
	Conv_3x3_Group,	// 3*3分组
	Conv_5x5_Group,	// 5*5分组
	Conv_3x3x3,		// 3输入通道3*3
};

struct stQuant {
	int flag = 0;
	int zero_point;
	int scale;
	float scale_inv;
};

struct stU31ConvData {
	int layer_idx;

	int kcount;
	int bcount;

	short* kernels;
	int* biases;

	enU31ConvType type;
	int split = 1;
};

static void dump_layer_bin(FILE* fp, const stU31ConvData& c)
{
	const short* sptr = c.kernels;
	const int* iptr = c.biases;
	int kcount = c.kcount / c.split;
	int bcount = c.bcount / c.split;
	int i;
	for (i = 0; i < c.split; i++) {
		fwrite(sptr, sizeof(short), kcount, fp);
		fwrite(iptr, sizeof(int), bcount, fp);		
		sptr += kcount;
		iptr += bcount;
	}
}

static void dump_bin(const char* filename, const std::vector<stU31ConvData>& convdata)
{
	FILE* fp = fopen(filename, "wb");
	for (auto& c : convdata) {
		dump_layer_bin(fp, c);
	}
	fclose(fp);
}

static std::vector<int> dump_bin_quant(
	const char* filename,
	const std::vector<stU31ConvData>& convdata,
	S16Network* net)
{
	std::vector<int> layer_sizes;
	std::map<int, const stU31ConvData*> cmap;
	for (auto& c : convdata) {
		cmap.insert({ c.layer_idx, &c });
	}

	FILE* fp = fopen(filename, "wb");
	for (auto layer : net->layers()) {
		int size = 0;
		auto it = cmap.find(layer->idx);
		if (it != cmap.end()) {
			size += it->second->kcount * 2 + it->second->bcount * 4;
			dump_layer_bin(fp, *it->second);
		}

		for (int i = 0; i < layer->input.quant_count; i++) {
			fwrite(&layer->input.quants[i].scale, sizeof(int), 1, fp);
			fwrite(&layer->input.quants[i].zero_point, sizeof(int), 1, fp);
			size += 8;
		}
		float finv = 1.0f / layer->output.quants[0].scale;
		fwrite(&finv, sizeof(float), 1, fp);
		fwrite(&layer->output.quants[0].zero_point, sizeof(int), 1, fp);
		size += 8;
		layer_sizes.push_back(size);
	}
	fclose(fp);
	return layer_sizes;
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

static void dump_shorts_3x3x3(FILE* fp, const short* sptr, int count)
{
	int i;
	fprintf(fp, ".short ");
	for (i = 0; i < count; i++, sptr++) {
		fprintf(fp, "0x%04x ", (unsigned short)(*sptr));
		if ((i + 1) % 28 % 8 == 0 || (i + 1) % 28 == 0) {
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


static void dump_layer_upasm(FILE* fp, const stU31ConvData& c)
{
	fprintf(fp, ".align 16\n");
	fprintf(fp, "//---------LAYER%d---------\n", c.layer_idx);
	if (c.split == 1) {
		fprintf(fp, "@export layer%d_kernel\n", c.layer_idx);
		printf("@import layer%d_kernel\n", c.layer_idx);
		fprintf(fp, "layer%d_kernel:\n", c.layer_idx);
		const short* sptr = c.kernels;
		if (c.type == Conv_3x3x3) {
			dump_shorts_3x3x3(fp, sptr, c.kcount);
		}
		else {
			dump_shorts(fp, sptr, c.kcount);
		}
		fprintf(fp, "@export layer%d_bias\n", c.layer_idx);
		fprintf(fp, "layer%d_bias:\n", c.layer_idx);
		dump_int(fp, c.biases, c.bcount);
		fprintf(fp, "\n");
	}
	else {
		assert(c.type == Conv_1x1);
		const short* sptr = c.kernels;
		const int* s32ptr = c.biases;
		int kcount = c.kcount / c.split;
		int bcount = c.bcount / c.split;
		int i;
		for (i = 0; i < c.split; i++) {
			fprintf(fp, "@export layer%d_kernel_p%d\n", c.layer_idx, i);
			printf("@import layer%d_kernel_p%d\n", c.layer_idx, i);
			fprintf(fp, "layer%d_kernel_p%d:\n", c.layer_idx, i);
			dump_shorts(fp, sptr, kcount);
			sptr += kcount;
			fprintf(fp, "@export layer%d_bias_p%d\n", c.layer_idx, i);
			fprintf(fp, "layer%d_bias_p%d:\n", c.layer_idx, i);
			dump_int(fp, s32ptr, bcount);
			s32ptr += bcount;
		}
		fprintf(fp, "\n");
	}
}

static void dump_upasm(const char* filename, const std::vector<stU31ConvData>& convdata)
{
	FILE* fp = fopen(filename, "wt");
	if (fp) {
		for (auto& c : convdata) {
			dump_layer_upasm(fp, c);
		}
		fclose(fp);
	}
}

static void dump_upasm_quant(
	const char* filename,
	const std::vector<stU31ConvData>& convdata,
	S16Network* net)
{
	std::vector<int> layer_sizes;
	std::map<int, const stU31ConvData*> cmap;
	for (auto& c : convdata) {
		cmap.insert({ c.layer_idx, &c });
	}

	FILE* fp = fopen(filename, "wt");
	if (fp) {
		for (auto layer : net->layers()) {
			int size = 0;
			auto it = cmap.find(layer->idx);
			if (it != cmap.end()) {
				dump_layer_upasm(fp, *it->second);
			}

			if (layer->input.quant_count) {
				fprintf(fp, "@export layer%d_input_quant\n", layer->idx);
				printf("@import layer%d_input_quant\n", layer->idx);
				for (int i = 0; i < layer->input.quant_count; i++) {
					fprintf(fp, ".word %d, %d\n",
						layer->input.quants[i].scale,
						layer->input.quants[i].zero_point);
				}
			}
			fprintf(fp, "@export layer%d_output_quant\n", layer->idx);
			printf("@import layer%d_output_quant\n", layer->idx);
			float finv = 1.0f / layer->output.quants[0].scale;
			fprintf(fp, ".word %g, %d\n",
				finv,
				layer->output.quants[0].zero_point);
			fprintf(fp, "\n");

		}
		fclose(fp);
	}

}


static void mergeConvData(stU31ConvData* c0, stU31ConvData* c1)
{
	int* tmpBias = (int*)malloc(sizeof(int) * c0->bcount);
	assert(tmpBias);
	assert((short*)(c0->biases + c0->bcount) == c1->kernels);

	memcpy(tmpBias, c0->biases, sizeof(int) * c0->bcount);
	memmove(c0->biases, c1->kernels, sizeof(short) * c1->kcount);
	c0->biases = (int*)(c0->kernels + c0->kcount + c1->kcount);
	memcpy(c0->biases, tmpBias, sizeof(int) * c0->bcount);
	c0->kcount += c1->kcount;
	c0->bcount += c1->bcount;
	c1->kcount = 0;
	c1->bcount = 0;

	free(tmpBias);
}

static std::vector<stU31ConvData> makeU31Data(const S16Network* network)
{
	std::vector<stU31ConvData> convdata;
	// 计算bin文件需要的数据个数
	int ktotal = 0, btotal = 0;
	for (auto layer : network->layers()) {
		if (layer->type == S16Layer_Convolution) {
			stU31ConvData data;
			S16ConvLayer* conv = (S16ConvLayer*)layer->priv;
			data.layer_idx = layer->idx;
			data.kcount = conv->ksize * conv->ksize * layer->input.c * layer->output.c / conv->groups;
			data.bcount = layer->output.c;			
			data.split = 1;

			if (conv->ksize == 3) {
				if (conv->groups == layer->output.c || layer->input.c == 1) {
					data.kcount = 2 * 8 * layer->input.c * layer->output.c / conv->groups;
					data.type = Conv_3x3_Group;
					///	// 3*3=9
					///	// 1个128bit寄存器可以存8个, 剩下1个也存1个128bit寄存器
					///	// 需要添7个0(16bit)
					///	data.kcount = 16 * layer->output.c;					
				}
				else if (layer->input.c == 3) {
					// 3*3*3 = 27, 
					// 3个128bit寄存器可以存3*8=24个, 剩下3个存1个64bit寄存器
					// 需要添1个0(16bit)
					data.kcount = 28 * layer->output.c;
					data.type = Conv_3x3x3;
				}
				else {
					assert(0);
				}
			}
			else if (conv->ksize == 5) {
				if (conv->groups == layer->output.c) {
					// 5*5=25
					// 3个128bit寄存器可以存24个, 剩下1个也存1个128bit寄存器
					// 需要添7个0(16bit)
					data.kcount = 32 * layer->input.c * layer->output.c / conv->groups;
					data.type = Conv_5x5_Group;
				}
				else {
					assert(0);
				}
			}
			else if (conv->ksize == 1) {
				data.type = Conv_1x1;
				if (layer->input.c >= 20) {
					int n = layer->output.c / layer->input.c;
					if (n >= 2 && layer->input.c * n == layer->output.c) {
						data.split = n;
					}
				}
			}
			else {
				assert(0);
			}

			ktotal += data.kcount;
			btotal += data.bcount;
			convdata.push_back(data);
		}
	}

	// 建立cmap
	//std::map<int, stU31ConvData*> cmap;
	//for (auto& c : convdata) {
	//	cmap.insert({ c.layer_idx, &c });
	//}

	auto& layerMap = network->layerMap();
	// 申请并分配内存
	short* data = (short*)malloc(sizeof(short) * ktotal + sizeof(int) * btotal);
	assert(data);
	short* sptr = data;
	for (auto& c : convdata) {
		c.kernels = sptr;
		c.biases = (int*)(c.kernels + c.kcount);
		sptr = (short*)(c.biases + c.bcount);

		auto it = layerMap.find(c.layer_idx);
		assert(it != layerMap.end());

		S16ConvLayer* conv = (S16ConvLayer*)it->second->priv;
		int channels = it->second->input.c * it->second->output.c / conv->groups;
		int kcount = conv->ksize * conv->ksize * channels;
		const short* psrc = conv->kernels;
		short* pdst = c.kernels;
		

		switch (c.type) {
		case Conv_1x1:
			assert(kcount == c.kcount);
			memcpy(c.kernels, conv->kernels, sizeof(short) * kcount);
			break;

		case Conv_3x3x3:
			for (int i = 0; i < it->second->output.c; i++) {
				memset(pdst, 0, sizeof(short) * 28);
				memcpy(pdst, psrc, sizeof(short) * 27);
				psrc += 27;
				pdst += 28;
			}
			break;

		case Conv_3x3_Group:
			for (int i = 0; i < channels; i++) {
				memset(pdst, 0, sizeof(short)* 16);
				memcpy(pdst, psrc, sizeof(short) * 9);
				psrc += 9;
				pdst += 16;
			}

		case Conv_5x5_Group:
			for (int i = 0; i < channels; i++) {
				memset(pdst, 0, sizeof(short)* 32);
				memcpy(pdst, psrc, sizeof(short) * 25);
				
				psrc += 25;
				pdst += 32;
			}
			break;
		}
		memcpy(c.biases, conv->biases, sizeof(int) * c.bcount);
	}

	// // 合并(113,115)(98,100)
	// mergeConvData(cmap[98], cmap[100]);
	// mergeConvData(cmap[113], cmap[115]);
	// // 移除100, 115
	// std::vector<stU31ConvData> tmpData;
	// convdata.clear();
	// for (auto& c : convdata) {
	// 	if (c.layer_idx != 100 && c.layer_idx != 115) {
	// 		tmpData.push_back(c);
	// 	}
	// }
	// convdata = tmpData;

	return convdata;
}


struct stGroup {
	int group_id;
	std::vector<int> layer_idx;
	int param_size = 0;
	int param_psram_addr = 0;
};

static const stGroup GROUPS[] = {
	{-1, {0}},
	{0, {1, 6, 7, 8}},
	{1, {2, 3, 4}},
	{2, {10, 11, 12, 13}},
	{3, {15, 16, 17, 18}},
	{4, {20, 21, 22, 23}},
	{5, {25, 26, 27, 29, 30, 31}},
	{6, {33, 34, 35, 36}},
	{7, {38, 39, 40, 41}},
	{8, {43, 44, 45, 46}},
	{9, {48, 49, 50, 51}},
	{10, {53, 54, 55, 56}},
	{11, {58, 59, 60, 61}},
	{12, {63, 64, 65, 66}},
	{13, {68, 69, 70, 72, 73, 74}},
	{14, {76, 77, 78, 79}},
	{15, {81, 82, 83, 84}},
	{16, {86, 87, 88, 89}},
	{17, {90, 93, 108, }},
	{18, {109, 110, 111, 112, 113, 117, 118, 119, 120, 121}},
	{19, {94, 95, 96, 97, 98, 102, 103, 104, 105, 106}}
};


static void calcGroupParams(const std::map<int, stU31ConvData*>& cmap)
{
	int group_count = sizeof(GROUPS) / sizeof(stGroup);
	int i;
	for (i = 0; i < group_count; i++) {
		int gsize = 0;
		for (auto idx : GROUPS[i].layer_idx) {
			auto it = cmap.find(idx);
			if (it != cmap.end()) {
				gsize += it->second->kcount * sizeof(short) + it->second->bcount * sizeof(int);
			}
		}
		printf("Group %d param size:%d\n", GROUPS[i].group_id, gsize);
	}
}

static void dump_layer_io(const S16Network* network)
{
	for (auto layer : network->layers()) {
		printf("#define LAYER_%d_INPUT_SIZE %d\n", layer->idx, layer->input.whc * 2);
		printf("#define LAYER_%d_OUTPUT_SIZE %d\n\n", layer->idx, layer->output.whc * 2);
	}
}

static void dump_params_addr(const std::vector<stU31ConvData>& convdata)
{
	printf("#define LAYER_0_KERNEL_ADDR 0\n");
	printf("#define LAYER_0_BIAS_ADDR LAYER_0_KERNEL_ADDR + LAYER_0_KERNEL_SIZE\n\n");
	for (int i = 1; i < (int)convdata.size(); i++) {
		int prev = convdata[i - 1].layer_idx;
		int curr = convdata[i].layer_idx;
		printf("#define LAYER_%d_KERNEL_ADDR LAYER_%d_BIAS_ADDR + LAYER_%d_BIAS_SIZE\n", curr, prev, prev);
		printf("#define LAYER_%d_BIAS_ADDR LAYER_%d_KERNEL_ADDR + LAYER_%d_KERNEL_SIZE\n\n", curr, curr, curr);
	}
}

static void dump_params_psram_addr(const std::vector<stU31ConvData>& convdata)
{
	int addr = 0;
	for (auto& c : convdata) {

		int ksize = c.kcount * sizeof(short);
		int bsize = c.bcount * sizeof(int);

		printf("#define LAYER_%d_PARAM_PSRAM_ADDR 0x%08x\n", c.layer_idx, addr);
		printf("#define LAYER_%d_KERNEL_SIZE %d\n", c.layer_idx, ksize);
		printf("#define LAYER_%d_BIAS_SIZE %d\n", c.layer_idx, bsize);
		printf("#define LAYER_%d_PARAM_SIZE %d\n", c.layer_idx, bsize + ksize);
		printf("\n");

		addr += bsize + ksize;
	}
}

static void printFlashAddr(int start, const std::vector<stU31ConvData>& convData)
{
	int addr = start;
	for (auto& c : convData) {
		int csize = c.kcount * 2 + c.bcount * 4;
		printf("#define LAYER_%d_KERNEL_FLASH_ADDR 0x%x\n", c.layer_idx, addr);
		printf("#define LAYER_%d_KERNEL_LENGTH %d // bytes\n", c.layer_idx, csize);
		addr += csize;		
	}
}

static void printFlashAddr(int start, const std::vector<int>& layer_sizes)
{
	int addr = start;
	for (int i = 0; i < (int)layer_sizes.size(); i++) {
		printf("#define LAYER_%d_KERNEL_FLASH_ADDR 0x%x\n", i, addr);
		printf("#define LAYER_%d_KERNEL_LENGTH %d // bytes\n", i, layer_sizes[i]);
		addr += layer_sizes[i];
	}
}

static void printWeightsLength(const std::vector<stU31ConvData>& convData)
{
	for (auto& c : convData) {
		int csize = (c.kcount * 2 + c.bcount * 4) / c.split;
		if (c.split == 1) {
			printf("#define LAYER_%d_KERNEL_LENGTH %d\n", c.layer_idx, csize);
		}
		else {
			printf("#define LAYER_%d_KERNEL_LENGTH %d\n", c.layer_idx, csize * c.split);
			printf("#define LAYER_%d_KERNEL_PART_LENGTH %d\n", c.layer_idx, csize);
		}
	}
}

static void printRAMAddr(const std::vector<stU31ConvData>& convData)
{
	int prev = -1;
	for (auto& c : convData) {
		printf("#define LAYER_%d_KERNEL_ADDR ", c.layer_idx);
		if (prev >= 0) {
			printf("(LAYER_%d_KERNEL_ADDR + LAYER_%d_KERNEL_LENGTH)\n", prev, prev);
		}
		else {
			printf("\n");
		}
		prev = c.layer_idx;
	}
}

static void printLayerOutputSize(const std::vector<S16BaseLayer*>& layers, int flag_8bit)
{
	for (auto& c : layers) {
		if (flag_8bit && (c->output.quants[0].scale != 1 && c->output.quants[0].zero_point != 0)) {
			printf("#define LAYER_%d_OUTPUT_SIZE %d // %d * %d * %d\n",
				c->idx, c->output.w * c->output.h * c->output.c,
				c->output.w, c->output.h, c->output.c);
		}
		else {
			printf("#define LAYER_%d_OUTPUT_SIZE %d // %d * %d * %d * 2\n", 
				c->idx, c->output.w * c->output.h * c->output.c * 2,
				c->output.w, c->output.h, c->output.c);
		}
	}
}

static void dump_quan_upasm(const char* filename, S16Network* net)
{
	FILE* fp = fopen(filename, "wt");
	if (fp) {
		for (auto layer : net->layers()) {
			fprintf(fp, "//-------layer%d----------\n", layer->idx);
			fprintf(fp, "layer%d_input_quant:\n", layer->idx);
			for (int i = 0; i < layer->input.quant_count; i++) {
				fprintf(fp, ".word %d, %d\n", layer->input.quants[i].scale, layer->input.quants[i].zero_point);
			}
			fprintf(fp, "layer%d_output_quant:\n", layer->idx);
			fprintf(fp, ".word %g, %d\n", 1.0f / layer->output.quants[0].scale, layer->output.quants[0].zero_point);
		}
		fclose(fp);
	}
}





int main(int argc, char** argv)
{
	S16Network* network = new S16Network;
	if (network->loadNetwork("../data/320x160.network.csv", 1)
		&& network->loadWeight("../data/320x160.weights.txt"))
	{
		auto scales = load_scales("../data/320x160_quant_scale.txt");
		set_scales(network, scales);

		//quantlize_weights(network,
		//	"../data/320x160.weights.txt",
		//	"320x160_maxmin.txt",
		//	"");
		//printLayerOutputSize(network->layers(), 1);

		std::vector<stU31ConvData> convdata = makeU31Data(network);


		//printRAMAddr(convdata);
		//printWeightsLength(convdata);
		//printFlashAddr(0x44000, convdata);
		//auto layer_sizes = dump_bin_quant("320x160_weights.bin", convdata, network);
		//printFlashAddr(0x44000, layer_sizes);
		dump_upasm_quant("320x160_weights.upasm", convdata, network);
		
		//for (auto layer : network->layers()) {
		//	bool found = false;
		//	for (auto& c : convdata) {
		//		if (c.layer_idx == layer->idx) {
		//			int weightSize = c.kcount * 2 + c.bcount * 4;
		//			int tt = weightSize * 40281 / 1024;
		//			printf("layer %2d, %5d,  %d\n", c.layer_idx, weightSize, tt);
		//			found = true;
		//			break;
		//		}
		//	}
		//	if (!found) {
		//		printf("layer %2d,     0,  0\n", layer->idx);
		//	}
		//	if (layer->idx == 48 || layer->idx == 64) {
		//		printf("解析,         0,  0\n");
		//	}
		//}

		int offset = 0;
		for (auto& c : convdata) {
			//std::string name = "layer_weights_bin/layer_" + std::to_string(c.layer_idx) + "_weights.bin";
			//FILE* fp = fopen(name.c_str(), "wb");
			//if (fp) {
			//	dump_layer_bin(fp, c);
			//	fclose(fp);
			//}

			//std::string name = "layer_weights/layer_" + std::to_string(c.layer_idx) + "_weights.upasm";
			//FILE* fp = fopen(name.c_str(), "wt");
			//if (fp) {
			//	dump_layer_upasm(fp, c);
			//	fclose(fp);
			//}
		}

		free(convdata[0].kernels);
	}

	delete network;

	return 0;
}