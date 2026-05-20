#include <assert.h>
#include <math.h>
#include <memory.h>
#include <stdlib.h>
#include "../u31/u31_inst.h"
#include "S16ConvLayer.h"

static void addbiases_linear(int *ptemp, s16 *pout, int biases, int n, int shift)
{
	int i;
	for (i = 0; i < n; i++, ptemp++, pout++) {
		int res = (*ptemp + biases) >> shift;
		if (res > 0x7fff) {
			res = 0x7fff;
		}
		else if (res < -0x8000) {
			//int t = *pout * 102;
			//*pout = t >> 10;
			res = 0x8000;
		}
		*pout = (s16)res;
	}
}

static void addbiases_leaky(int* ptemp, s16* pout, int biases, int n, int shift)
{
	int i;
	for (i = 0; i < n; i++, ptemp++, pout++) {
		int res = (*ptemp + biases) >> shift;
		if (res > 0x7fff) {
			res = 0x7fff;
		}
		else if (res < 0) {
			//int t = *pout * 102;
			//*pout = t >> 10;
			res >>= 4;
		}
		*pout = (s16)res;
	}
}

static void addbiases_relu(int* ptemp, s16* pout, int biases, int n, int shift)
{
	int i;
	for (i = 0; i < n; i++, ptemp++, pout++) {
		int res = (*ptemp + biases) >> shift;
		if (res > 0x7fff) {
			res = 0x7fff;
		}
		else if (res < 0) {
			//int t = *pout * 102;
			//*pout = t >> 10;
			res = 0;
		}
		*pout = (s16)res;
	}
}

static void addbiases_sigmoid(int* ptemp, s16* pout, int biases, int n, int shift)
{
	int i;
	for (i = 0; i < n; i++, ptemp++, pout++) {
		*pout = u31_sigmoid((*ptemp + biases) >> shift);
	}
}


typedef void (*addbiase_activate_func_t)(int* ptemp, s16* pout, int biases, int n, int shift);

#define KXI(a) ((int)pkernel[(a)] * pin[idx_##a])
//#define KXI(a) ((pkernel[(a)] * pin[idx_##a]) >> S16_QUANT_SHIFT)

static void add_conv_k3s2(const s16 *pin, const s16 *pkernel, int *pout, int w, int h, int pad)
{
	const s16 *pstart = pin;
	int i, j;
	const int idx_0 = -w - 1;
	const int idx_1 = -w;
	const int idx_2 = -w + 1;
	const int idx_3 = -1;
	const int idx_4 = 0;
	const int idx_5 = +1;
	const int idx_6 = w - 1;
	const int idx_7 = w;
	const int idx_8 = w + 1;
	assert(pad == 1);
	
	// 第一行
	*pout += KXI(4) + KXI(5) + KXI(7) + KXI(8);
	pin += 2; 
	pout++;

	for (i = 2; i < w; i += 2, pin += 2, pout++) {
		*pout += KXI(3) + KXI(4) + KXI(5) + KXI(6) + KXI(7) + KXI(8);
	}

	// 中间行
	pstart += w * 2;
	for (i = 2; i < h; i += 2, pstart += w*2) {
		pin = pstart;

		*pout += KXI(1) + KXI(2) + KXI(4) + KXI(5) + KXI(7) + KXI(8);
		pin += 2; 
		pout++;

		for (j = 2; j < w; j += 2, pin += 2, pout++) {
			*pout += KXI(0) + KXI(1) + KXI(2) + KXI(3) + KXI(4) + KXI(5) + KXI(6) + KXI(7) + KXI(8);
		}
	}
}

static void add_conv_k5s1p2(const s16* pin, const s16* pkernel, int* pout, int w, int h)
{
	const s16* pstart = pin;
	int i, j;
	const int idx_0 = -w * 2 - 2;
	const int idx_1 = -w * 2 - 1;
	const int idx_2 = -w * 2;
	const int idx_3 = -w * 2 + 1;
	const int idx_4 = -w * 2 + 2;
	const int idx_5 = -w * 1 - 2;
	const int idx_6 = -w * 1 - 1;
	const int idx_7 = -w * 1;
	const int idx_8 = -w * 1 + 1;
	const int idx_9 = -w * 1 + 2;
	const int idx_10 =  - 2;
	const int idx_11 =  - 1;
	const int idx_12 = 0;
	const int idx_13 =  + 1;
	const int idx_14 =  + 2;
	const int idx_15 = w * 1 - 2;
	const int idx_16 = w * 1 - 1;
	const int idx_17 = w * 1;
	const int idx_18 = w * 1 + 1;
	const int idx_19 = w * 1 + 2;
	const int idx_20 = w * 2 - 2;
	const int idx_21 = w * 2 - 1;
	const int idx_22 = w * 2;
	const int idx_23 = w * 2 + 1;
	const int idx_24 = w * 2 + 2;

	/*
		0	1	2	3	4
		5	6	7	8	9
		10	11	12	13	14
		15	16	17	18	19
		20	21	22	23	24	
	*/

	// 第一行
	*pout += KXI(12) + KXI(13) + KXI(14);
	*pout += KXI(17) + KXI(18) + KXI(19);
	*pout += KXI(22) + KXI(23) + KXI(24);
	pin++; pout++;

	*pout += KXI(11) + KXI(12) + KXI(13) + KXI(14);
	*pout += KXI(16) + KXI(17) + KXI(18) + KXI(19);
	*pout += KXI(21) + KXI(22) + KXI(23) + KXI(24);
	pin++; pout++;

	for (i = 2; i < w - 2; i++, pin++, pout++) {
		*pout += KXI(10) + KXI(11) + KXI(12) + KXI(13) + KXI(14);
		*pout += KXI(15) + KXI(16) + KXI(17) + KXI(18) + KXI(19);
		*pout += KXI(20) + KXI(21) + KXI(22) + KXI(23) + KXI(24);
	}

	*pout += KXI(10) + KXI(11) + KXI(12) + KXI(13);
	*pout += KXI(15) + KXI(16) + KXI(17) + KXI(18);
	*pout += KXI(20) + KXI(21) + KXI(22) + KXI(23);
	pin++; pout++;

	*pout += KXI(10) + KXI(11) + KXI(12);
	*pout += KXI(15) + KXI(16) + KXI(17);
	*pout += KXI(20) + KXI(21) + KXI(22);
	pin++; pout++;

	// 第二行
	*pout += KXI( 7) + KXI( 8) + KXI( 9);
	*pout += KXI(12) + KXI(13) + KXI(14);
	*pout += KXI(17) + KXI(18) + KXI(19);
	*pout += KXI(22) + KXI(23) + KXI(24);
	pin++; pout++;

	*pout += KXI( 6) + KXI( 7) + KXI( 8) + KXI( 9);
	*pout += KXI(11) + KXI(12) + KXI(13) + KXI(14);
	*pout += KXI(16) + KXI(17) + KXI(18) + KXI(19);
	*pout += KXI(21) + KXI(22) + KXI(23) + KXI(24);
	pin++; pout++;

	for (i = 2; i < w - 2; i++, pin++, pout++) {
		*pout += KXI( 5) + KXI( 6) + KXI( 7) + KXI( 8) + KXI( 9);
		*pout += KXI(10) + KXI(11) + KXI(12) + KXI(13) + KXI(14);
		*pout += KXI(15) + KXI(16) + KXI(17) + KXI(18) + KXI(19);
		*pout += KXI(20) + KXI(21) + KXI(22) + KXI(23) + KXI(24);
	}
	*pout += KXI( 5) + KXI( 6) + KXI( 7) + KXI( 8);
	*pout += KXI(10) + KXI(11) + KXI(12) + KXI(13);
	*pout += KXI(15) + KXI(16) + KXI(17) + KXI(18);
	*pout += KXI(20) + KXI(21) + KXI(22) + KXI(23);
	pin++; pout++;

	*pout += KXI( 5) + KXI( 6) + KXI( 7);
	*pout += KXI(10) + KXI(11) + KXI(12);
	*pout += KXI(15) + KXI(16) + KXI(17);
	*pout += KXI(20) + KXI(21) + KXI(22);
	pin++; pout++;

	// 中间行
	pstart += w * 2;
	for (i = 2; i < h - 2; i++, pstart += w) {
		pin = pstart;

		*pout += KXI( 2) + KXI( 3) + KXI( 4);
		*pout += KXI( 7) + KXI( 8) + KXI( 9);
		*pout += KXI(12) + KXI(13) + KXI(14);
		*pout += KXI(17) + KXI(18) + KXI(19);
		*pout += KXI(22) + KXI(23) + KXI(24);
		pin++; pout++;

		*pout += KXI( 1) + KXI( 2) + KXI( 3) + KXI( 4);
		*pout += KXI( 6) + KXI( 7) + KXI( 8) + KXI( 9);
		*pout += KXI(11) + KXI(12) + KXI(13) + KXI(14);
		*pout += KXI(16) + KXI(17) + KXI(18) + KXI(19);
		*pout += KXI(21) + KXI(22) + KXI(23) + KXI(24);
		pin++; pout++;

		for (j = 2; j < w - 2; j++, pin++, pout++) {
			*pout += KXI( 0) + KXI( 1) + KXI( 2) + KXI( 3) + KXI( 4);
			*pout += KXI( 5) + KXI( 6) + KXI( 7) + KXI( 8) + KXI( 9);
			*pout += KXI(10) + KXI(11) + KXI(12) + KXI(13) + KXI(14);
			*pout += KXI(15) + KXI(16) + KXI(17) + KXI(18) + KXI(19);
			*pout += KXI(20) + KXI(21) + KXI(22) + KXI(23) + KXI(24);
		}

		*pout += KXI( 0) + KXI( 1) + KXI( 2) + KXI( 3);
		*pout += KXI( 5) + KXI( 6) + KXI( 7) + KXI( 8);
		*pout += KXI(10) + KXI(11) + KXI(12) + KXI(13);
		*pout += KXI(15) + KXI(16) + KXI(17) + KXI(18);
		*pout += KXI(20) + KXI(21) + KXI(22) + KXI(23);
		pin++; pout++;

		*pout += KXI( 0) + KXI( 1) + KXI( 2);
		*pout += KXI( 5) + KXI( 6) + KXI( 7);
		*pout += KXI(10) + KXI(11) + KXI(12);
		*pout += KXI(15) + KXI(16) + KXI(17);
		*pout += KXI(20) + KXI(21) + KXI(22);
		pin++; pout++;
	}

	*pout += KXI( 2) + KXI( 3) + KXI( 4);
	*pout += KXI( 7) + KXI( 8) + KXI( 9);
	*pout += KXI(12) + KXI(13) + KXI(14);
	*pout += KXI(17) + KXI(18) + KXI(19);
	pin++; pout++;

	*pout += KXI( 1) + KXI( 2) + KXI( 3) + KXI( 4);
	*pout += KXI( 6) + KXI( 7) + KXI( 8) + KXI( 9);
	*pout += KXI(11) + KXI(12) + KXI(13) + KXI(14);
	*pout += KXI(16) + KXI(17) + KXI(18) + KXI(19);
	pin++; pout++;

	for (j = 2; j < w - 2; j++, pin++, pout++) {
		*pout += KXI( 0) + KXI( 1) + KXI( 2) + KXI( 3) + KXI( 4);
		*pout += KXI( 5) + KXI( 6) + KXI( 7) + KXI( 8) + KXI( 9);
		*pout += KXI(10) + KXI(11) + KXI(12) + KXI(13) + KXI(14);
		*pout += KXI(15) + KXI(16) + KXI(17) + KXI(18) + KXI(19);
	}

	*pout += KXI( 0) + KXI( 1) + KXI( 2) + KXI( 3);
	*pout += KXI( 5) + KXI( 6) + KXI( 7) + KXI( 8);
	*pout += KXI(10) + KXI(11) + KXI(12) + KXI(13);
	*pout += KXI(15) + KXI(16) + KXI(17) + KXI(18);
	pin++; pout++;

	*pout += KXI( 0) + KXI( 1) + KXI( 2);
	*pout += KXI( 5) + KXI( 6) + KXI( 7);
	*pout += KXI(10) + KXI(11) + KXI(12);
	*pout += KXI(15) + KXI(16) + KXI(17);
	pin++; pout++;

	*pout += KXI( 2) + KXI( 3) + KXI( 4);
	*pout += KXI( 7) + KXI( 8) + KXI( 9);
	*pout += KXI(12) + KXI(13) + KXI(14);
	pin++; pout++;

	*pout += KXI( 1) + KXI( 2) + KXI( 3) + KXI( 4);
	*pout += KXI( 6) + KXI( 7) + KXI( 8) + KXI( 9);
	*pout += KXI(11) + KXI(12) + KXI(13) + KXI(14);
	pin++; pout++;

	for (j = 2; j < w - 2; j++, pin++, pout++) {
		*pout += KXI( 0) + KXI( 1) + KXI( 2) + KXI( 3) + KXI( 4);
		*pout += KXI( 5) + KXI( 6) + KXI( 7) + KXI( 8) + KXI( 9);
		*pout += KXI(10) + KXI(11) + KXI(12) + KXI(13) + KXI(14);
	}

	*pout += KXI( 0) + KXI( 1) + KXI( 2) + KXI( 3);
	*pout += KXI( 5) + KXI( 6) + KXI( 7) + KXI( 8);
	*pout += KXI(10) + KXI(11) + KXI(12) + KXI(13);
	pin++; pout++;

	*pout += KXI( 0) + KXI( 1) + KXI( 2);
	*pout += KXI( 5) + KXI( 6) + KXI( 7);
	*pout += KXI(10) + KXI(11) + KXI(12);
	pin++; pout++;
}


static void add_conv_k3s1p1(const s16 *pin, const s16 *pkernel, int *pout, int w, int h, int pad)
{
	const s16 *pstart = pin;
	const int idx_0 = -w - 1;
	const int idx_1 = -w;
	const int idx_2 = -w + 1;
	const int idx_3 = -1;
	const int idx_4 = 0;
	const int idx_5 = +1;
	const int idx_6 = w - 1;
	const int idx_7 = w;
	const int idx_8 = w + 1;
	int i, j;
	assert(pad == 1);

	// 第一行
	*pout += KXI(4) + KXI(5) + KXI(7) + KXI(8);
	pin++; pout++;

	for (i = 1; i < w - 1; i++, pin++, pout++) {
		*pout += KXI(3) + KXI(4) + KXI(5) + KXI(6) + KXI(7) + KXI(8);
	}

	*pout += KXI(3) + KXI(4) + KXI(6) + KXI(7);
	pout++;

	// 中间行
	pstart += w;
	for (i = 1; i < h - 1; i++, pstart += w) {
		pin = pstart;

		*pout += KXI(1) + KXI(2) + KXI(4) + KXI(5) + KXI(7) + KXI(8);
		pin++; pout++;

		for (j = 1; j < w - 1; j++, pin++, pout++) {
			*pout += KXI(0) + KXI(1) + KXI(2) + KXI(3) + KXI(4) + KXI(5) + KXI(6) + KXI(7) + KXI(8);
		}

		*pout += KXI(0) + KXI(1) + KXI(3) + KXI(4) + KXI(6) + KXI(7);
		pout++;
	}

	// 最后一行
	pin = pstart;
	*pout += KXI(1) + KXI(2) + KXI(4) + KXI(5);
	pin++; pout++;

	for (i = 1; i < w - 1; i++, pin++, pout++) {
		*pout += KXI(0) + KXI(1) + KXI(2) + KXI(3) + KXI(4) + KXI(5);
	}

	*pout += KXI(0) + KXI(1) + KXI(3) + KXI(4);
}

static void add_conv_k3s1p0(const s16* pin, const s16* pkernel, int* pout, int w, int h, int pad)
{
	const s16* pstart = pin + w + 1;
	const int idx_0 = -w - 1;
	const int idx_1 = -w;
	const int idx_2 = -w + 1;
	const int idx_3 = -1;
	const int idx_4 = 0;
	const int idx_5 = +1;
	const int idx_6 = w - 1;
	const int idx_7 = w;
	const int idx_8 = w + 1;
	int i, j;
	assert(pad == 0);
	for (i = 0; i < h - 2; i++, pstart += w) {
		pin = pstart;
		for (j = 0; j < w - 2; j++, pin++, pout++) {
			*pout += KXI(0) + KXI(1) + KXI(2) + KXI(3) + KXI(4) + KXI(5) + KXI(6) + KXI(7) + KXI(8);
		}
	}
}

static void add_conv_k1(const s16 *pin, s16 kernel, int *pout, int count)
{
	int i;
	for (i = 0; i < count; i++, pin++, pout++) {
		*pout += (*pin * kernel);
	}
}

#if S16_QUANT_SHIFT < 8
static void shift_array(s16 *data, int count, int n)
{
	int i;
	for (i = 0; i < count; i++, data++) {
		*data >>= n;
	}
}
#endif

CAPI void S16_ConvolutionForward (void *userdata, const S16Tensor *input, S16Tensor *output) {
	S16ConvLayer *conv = (S16ConvLayer*)userdata;
	const int in_count = input->w * input->h;
	const int out_count = output->w * output->h;

	int i, j, k;
	int* pbiase = conv->biases;
	s16 *pin, *pout = output->ptr, *pkernel = conv->kernels;
	int* temp = (int*)malloc(sizeof(int) * out_count);
	addbiase_activate_func_t biase_func = NULL;
	//assert(conv->ksize == 3);
	memset(output->ptr, 0, sizeof(s16)*output->whc);

	switch (conv->activate_type) {
	case ActivateType_Linear: biase_func = addbiases_linear; break;
	case ActivateType_Leaky: biase_func = addbiases_leaky; break;
	case ActivateType_Relu: biase_func = addbiases_relu; break;
	case ActivateType_Sigmoid: biase_func = addbiases_sigmoid; break;
	default: assert(0);
	}

	if (conv->ksize == 5) {
		assert(conv->stride == 1 && conv->pad == 2);
		assert(conv->groups == output->c); // dw
		assert(input->w == output->w);
		assert(input->h == output->h);
		assert(input->c == output->c);

		for (i = 0; i < conv->groups; i++) {
			pout = output->ptr + i * (output->whc / conv->groups);
			for (j = 0; j < output->c / conv->groups; j++, pout += out_count, pbiase++) {
				memset(temp, 0, sizeof(int) * out_count);
				pin = input->ptr + i * (input->whc / conv->groups);
				for (k = 0; k < input->c / conv->groups; k++, pin += in_count, pkernel += 25) {
					add_conv_k5s1p2(pin, pkernel, temp, input->w, input->h);
				}
				biase_func(temp, pout, *pbiase, out_count, conv->shift);
			}
		}
	}
	else if (conv->ksize == 3) {
		if (conv->stride == 2 && input->w % 2 == 0 && input->h % 2 == 0) {
			for (i = 0; i < conv->groups; i++) {
				pout = output->ptr + i * (output->whc / conv->groups);				
				for (j = 0; j < output->c / conv->groups; j++, pout += out_count, pbiase++) {
					memset(temp, 0, sizeof(int) * out_count);
					pin = input->ptr + i * (input->whc / conv->groups);
					for (k = 0; k < input->c / conv->groups; k++, pin += in_count, pkernel += 9) {
						add_conv_k3s2(pin, pkernel, temp, input->w, input->h, conv->pad);
					}
					biase_func(temp, pout, *pbiase, out_count, conv->shift);
				}
			}
		}
		else if (conv->stride == 1) {
			if (conv->pad == 1) {
				for (i = 0; i < conv->groups; i++) {
					pout = output->ptr + i * (output->whc / conv->groups);
					for (j = 0; j < output->c / conv->groups; j++, pout += out_count, pbiase++) {
						memset(temp, 0, sizeof(int) * out_count);
						pin = input->ptr + i * (input->whc / conv->groups);
						for (k = 0; k < input->c / conv->groups; k++, pin += in_count, pkernel += 9) {
							add_conv_k3s1p1(pin, pkernel, temp, input->w, input->h, conv->pad);
						}
						biase_func(temp, pout, *pbiase, out_count, conv->shift);
					}
				}
			}
			else if (conv->pad == 0) {
				for (i = 0; i < conv->groups; i++) {
					pout = output->ptr + i * (output->whc / conv->groups);
					for (j = 0; j < output->c / conv->groups; j++, pout += out_count, pbiase++) {
						memset(temp, 0, sizeof(int) * out_count);
						pin = input->ptr + i * (input->whc / conv->groups);
						for (k = 0; k < input->c / conv->groups; k++, pin += in_count, pkernel += 9) {
							add_conv_k3s1p0(pin, pkernel, temp, input->w, input->h, conv->pad);
						}
						biase_func(temp, pout, *pbiase, out_count, conv->shift);
					}
				}
			}
		}
		else {
			assert(0);
		}
	}
	else if (conv->ksize == 1) {
		for (i = 0; i < conv->groups; i++) {
			pout = output->ptr + i * (output->whc / conv->groups);
			for (j = 0; j < output->c / conv->groups; j++, pout += out_count, pbiase++) {
				memset(temp, 0, sizeof(int) * out_count);
				pin = input->ptr + i * (input->whc / conv->groups);
				for (k = 0; k < input->c / conv->groups; k++, pin += in_count, pkernel++) {
					add_conv_k1(pin, *pkernel, temp, in_count);
				}
				biase_func(temp, pout, *pbiase, out_count, conv->shift);
			}
		}
	}
	free(temp);
}