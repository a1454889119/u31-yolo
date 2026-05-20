#include <assert.h>
#include <math.h>
#include <memory.h>
#include "UpixBaseLayer.h"

#define KXI(a) (pkernel[(a)] * pin[idx_##a])
template<typename T>
void add_conv_k3s2p1(const T* pin, const T* pkernel, T* pout, T padv, int w, int h, int pad)
{
	const T* pstart = pin;
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

	T lt = padv * (pkernel[0] + pkernel[1] + pkernel[2] + pkernel[3] + pkernel[6]);
	T top = padv * (pkernel[0] + pkernel[1] + pkernel[2]);
	T left = padv * (pkernel[0] + pkernel[3] + pkernel[6]);

	// 第一行
	*pout += lt + KXI(4) + KXI(5) + KXI(7) + KXI(8);
	pin += 2;
	pout++;

	for (i = 2; i < w; i += 2, pin += 2, pout++) {
		*pout += top + KXI(3) + KXI(4) + KXI(5) + KXI(6) + KXI(7) + KXI(8);
	}

	// 中间行
	pstart += w * 2;
	for (i = 2; i < h; i += 2, pstart += w * 2) {
		pin = pstart;

		*pout += left + KXI(1) + KXI(2) + KXI(4) + KXI(5) + KXI(7) + KXI(8);
		pin += 2;
		pout++;

		for (j = 2; j < w; j += 2, pin += 2, pout++) {
			*pout += KXI(0) + KXI(1) + KXI(2) + KXI(3) + KXI(4) + KXI(5) + KXI(6) + KXI(7) + KXI(8);
		}
	}
}

//template<typename T>
//void add_conv_k5s1p2(const T* pin, const T* pkernel, T* pout, T padv, int w, int h)
//{
//	const T* pstart = pin;
//	int i, j;
//	const int idx_0 = -w * 2 - 2;
//	const int idx_1 = -w * 2 - 1;
//	const int idx_2 = -w * 2;
//	const int idx_3 = -w * 2 + 1;
//	const int idx_4 = -w * 2 + 2;
//	const int idx_5 = -w * 1 - 2;
//	const int idx_6 = -w * 1 - 1;
//	const int idx_7 = -w * 1;
//	const int idx_8 = -w * 1 + 1;
//	const int idx_9 = -w * 1 + 2;
//	const int idx_10 = -2;
//	const int idx_11 = -1;
//	const int idx_12 = 0;
//	const int idx_13 = +1;
//	const int idx_14 = +2;
//	const int idx_15 = w * 1 - 2;
//	const int idx_16 = w * 1 - 1;
//	const int idx_17 = w * 1;
//	const int idx_18 = w * 1 + 1;
//	const int idx_19 = w * 1 + 2;
//	const int idx_20 = w * 2 - 2;
//	const int idx_21 = w * 2 - 1;
//	const int idx_22 = w * 2;
//	const int idx_23 = w * 2 + 1;
//	const int idx_24 = w * 2 + 2;
//
//	/*
//		0	1	2	3	4
//		5	6	7	8	9
//		10	11	12	13	14
//		15	16	17	18	19
//		20	21	22	23	24
//	*/
//
//	// 第一行
//	*pout += KXI(12) + KXI(13) + KXI(14);
//	*pout += KXI(17) + KXI(18) + KXI(19);
//	*pout += KXI(22) + KXI(23) + KXI(24);
//	pin++; pout++;
//
//	*pout += KXI(11) + KXI(12) + KXI(13) + KXI(14);
//	*pout += KXI(16) + KXI(17) + KXI(18) + KXI(19);
//	*pout += KXI(21) + KXI(22) + KXI(23) + KXI(24);
//	pin++; pout++;
//
//	for (i = 2; i < w - 2; i++, pin++, pout++) {
//		*pout += KXI(10) + KXI(11) + KXI(12) + KXI(13) + KXI(14);
//		*pout += KXI(15) + KXI(16) + KXI(17) + KXI(18) + KXI(19);
//		*pout += KXI(20) + KXI(21) + KXI(22) + KXI(23) + KXI(24);
//	}
//
//	*pout += KXI(10) + KXI(11) + KXI(12) + KXI(13);
//	*pout += KXI(15) + KXI(16) + KXI(17) + KXI(18);
//	*pout += KXI(20) + KXI(21) + KXI(22) + KXI(23);
//	pin++; pout++;
//
//	*pout += KXI(10) + KXI(11) + KXI(12);
//	*pout += KXI(15) + KXI(16) + KXI(17);
//	*pout += KXI(20) + KXI(21) + KXI(22);
//	pin++; pout++;
//
//	// 第二行
//	*pout += KXI(7) + KXI(8) + KXI(9);
//	*pout += KXI(12) + KXI(13) + KXI(14);
//	*pout += KXI(17) + KXI(18) + KXI(19);
//	*pout += KXI(22) + KXI(23) + KXI(24);
//	pin++; pout++;
//
//	*pout += KXI(6) + KXI(7) + KXI(8) + KXI(9);
//	*pout += KXI(11) + KXI(12) + KXI(13) + KXI(14);
//	*pout += KXI(16) + KXI(17) + KXI(18) + KXI(19);
//	*pout += KXI(21) + KXI(22) + KXI(23) + KXI(24);
//	pin++; pout++;
//
//	for (i = 2; i < w - 2; i++, pin++, pout++) {
//		*pout += KXI(5) + KXI(6) + KXI(7) + KXI(8) + KXI(9);
//		*pout += KXI(10) + KXI(11) + KXI(12) + KXI(13) + KXI(14);
//		*pout += KXI(15) + KXI(16) + KXI(17) + KXI(18) + KXI(19);
//		*pout += KXI(20) + KXI(21) + KXI(22) + KXI(23) + KXI(24);
//	}
//	*pout += KXI(5) + KXI(6) + KXI(7) + KXI(8);
//	*pout += KXI(10) + KXI(11) + KXI(12) + KXI(13);
//	*pout += KXI(15) + KXI(16) + KXI(17) + KXI(18);
//	*pout += KXI(20) + KXI(21) + KXI(22) + KXI(23);
//	pin++; pout++;
//
//	*pout += KXI(5) + KXI(6) + KXI(7);
//	*pout += KXI(10) + KXI(11) + KXI(12);
//	*pout += KXI(15) + KXI(16) + KXI(17);
//	*pout += KXI(20) + KXI(21) + KXI(22);
//	pin++; pout++;
//
//	// 中间行
//	pstart += w * 2;
//	for (i = 2; i < h - 2; i++, pstart += w) {
//		pin = pstart;
//
//		*pout += KXI(2) + KXI(3) + KXI(4);
//		*pout += KXI(7) + KXI(8) + KXI(9);
//		*pout += KXI(12) + KXI(13) + KXI(14);
//		*pout += KXI(17) + KXI(18) + KXI(19);
//		*pout += KXI(22) + KXI(23) + KXI(24);
//		pin++; pout++;
//
//		*pout += KXI(1) + KXI(2) + KXI(3) + KXI(4);
//		*pout += KXI(6) + KXI(7) + KXI(8) + KXI(9);
//		*pout += KXI(11) + KXI(12) + KXI(13) + KXI(14);
//		*pout += KXI(16) + KXI(17) + KXI(18) + KXI(19);
//		*pout += KXI(21) + KXI(22) + KXI(23) + KXI(24);
//		pin++; pout++;
//
//		for (j = 2; j < w - 2; j++, pin++, pout++) {
//			*pout += KXI(0) + KXI(1) + KXI(2) + KXI(3) + KXI(4);
//			*pout += KXI(5) + KXI(6) + KXI(7) + KXI(8) + KXI(9);
//			*pout += KXI(10) + KXI(11) + KXI(12) + KXI(13) + KXI(14);
//			*pout += KXI(15) + KXI(16) + KXI(17) + KXI(18) + KXI(19);
//			*pout += KXI(20) + KXI(21) + KXI(22) + KXI(23) + KXI(24);
//		}
//
//		*pout += KXI(0) + KXI(1) + KXI(2) + KXI(3);
//		*pout += KXI(5) + KXI(6) + KXI(7) + KXI(8);
//		*pout += KXI(10) + KXI(11) + KXI(12) + KXI(13);
//		*pout += KXI(15) + KXI(16) + KXI(17) + KXI(18);
//		*pout += KXI(20) + KXI(21) + KXI(22) + KXI(23);
//		pin++; pout++;
//
//		*pout += KXI(0) + KXI(1) + KXI(2);
//		*pout += KXI(5) + KXI(6) + KXI(7);
//		*pout += KXI(10) + KXI(11) + KXI(12);
//		*pout += KXI(15) + KXI(16) + KXI(17);
//		*pout += KXI(20) + KXI(21) + KXI(22);
//		pin++; pout++;
//	}
//
//	*pout += KXI(2) + KXI(3) + KXI(4);
//	*pout += KXI(7) + KXI(8) + KXI(9);
//	*pout += KXI(12) + KXI(13) + KXI(14);
//	*pout += KXI(17) + KXI(18) + KXI(19);
//	pin++; pout++;
//
//	*pout += KXI(1) + KXI(2) + KXI(3) + KXI(4);
//	*pout += KXI(6) + KXI(7) + KXI(8) + KXI(9);
//	*pout += KXI(11) + KXI(12) + KXI(13) + KXI(14);
//	*pout += KXI(16) + KXI(17) + KXI(18) + KXI(19);
//	pin++; pout++;
//
//	for (j = 2; j < w - 2; j++, pin++, pout++) {
//		*pout += KXI(0) + KXI(1) + KXI(2) + KXI(3) + KXI(4);
//		*pout += KXI(5) + KXI(6) + KXI(7) + KXI(8) + KXI(9);
//		*pout += KXI(10) + KXI(11) + KXI(12) + KXI(13) + KXI(14);
//		*pout += KXI(15) + KXI(16) + KXI(17) + KXI(18) + KXI(19);
//	}
//
//	*pout += KXI(0) + KXI(1) + KXI(2) + KXI(3);
//	*pout += KXI(5) + KXI(6) + KXI(7) + KXI(8);
//	*pout += KXI(10) + KXI(11) + KXI(12) + KXI(13);
//	*pout += KXI(15) + KXI(16) + KXI(17) + KXI(18);
//	pin++; pout++;
//
//	*pout += KXI(0) + KXI(1) + KXI(2);
//	*pout += KXI(5) + KXI(6) + KXI(7);
//	*pout += KXI(10) + KXI(11) + KXI(12);
//	*pout += KXI(15) + KXI(16) + KXI(17);
//	pin++; pout++;
//
//	*pout += KXI(2) + KXI(3) + KXI(4);
//	*pout += KXI(7) + KXI(8) + KXI(9);
//	*pout += KXI(12) + KXI(13) + KXI(14);
//	pin++; pout++;
//
//	*pout += KXI(1) + KXI(2) + KXI(3) + KXI(4);
//	*pout += KXI(6) + KXI(7) + KXI(8) + KXI(9);
//	*pout += KXI(11) + KXI(12) + KXI(13) + KXI(14);
//	pin++; pout++;
//
//	for (j = 2; j < w - 2; j++, pin++, pout++) {
//		*pout += KXI(0) + KXI(1) + KXI(2) + KXI(3) + KXI(4);
//		*pout += KXI(5) + KXI(6) + KXI(7) + KXI(8) + KXI(9);
//		*pout += KXI(10) + KXI(11) + KXI(12) + KXI(13) + KXI(14);
//	}
//
//	*pout += KXI(0) + KXI(1) + KXI(2) + KXI(3);
//	*pout += KXI(5) + KXI(6) + KXI(7) + KXI(8);
//	*pout += KXI(10) + KXI(11) + KXI(12) + KXI(13);
//	pin++; pout++;
//
//	*pout += KXI(0) + KXI(1) + KXI(2);
//	*pout += KXI(5) + KXI(6) + KXI(7);
//	*pout += KXI(10) + KXI(11) + KXI(12);
//	pin++; pout++;
//}

template<typename T>
void add_conv_k3s1p1(const T* pin, const T* pkernel, T* pout, T padv, int w, int h, int pad)
{
	const T* pstart = pin;
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

	T top = padv * (pkernel[0] + pkernel[1] + pkernel[2]);
	T bot = padv * (pkernel[6] + pkernel[7] + pkernel[8]);
	T left = padv * (pkernel[0] + pkernel[3] + pkernel[6]);
	T right = padv * (pkernel[2] + pkernel[5] + pkernel[8]);

	T lt = top + padv * (pkernel[3] + pkernel[6]);
	T rt = top + padv * (pkernel[5] + pkernel[8]);
	T lb = bot + padv * (pkernel[0] + pkernel[3]);
	T rb = bot + padv * (pkernel[2] + pkernel[5]);
	

	// 第一行
	*pout += lt +  KXI(4) + KXI(5) + KXI(7) + KXI(8);
	pin++; pout++;

	for (i = 1; i < w - 1; i++, pin++, pout++) {
		*pout += top + KXI(3) + KXI(4) + KXI(5) + KXI(6) + KXI(7) + KXI(8);
	}

	*pout += rt + KXI(3) + KXI(4) + KXI(6) + KXI(7);
	pout++;

	// 中间行
	pstart += w;
	for (i = 1; i < h - 1; i++, pstart += w) {
		pin = pstart;

		*pout += left + KXI(1) + KXI(2) + KXI(4) + KXI(5) + KXI(7) + KXI(8);
		pin++; pout++;

		for (j = 1; j < w - 1; j++, pin++, pout++) {
			*pout += KXI(0) + KXI(1) + KXI(2) + KXI(3) + KXI(4) + KXI(5) + KXI(6) + KXI(7) + KXI(8);
		}

		*pout += right + KXI(0) + KXI(1) + KXI(3) + KXI(4) + KXI(6) + KXI(7);
		pout++;
	}

	// 最后一行
	pin = pstart;
	*pout += lb + KXI(1) + KXI(2) + KXI(4) + KXI(5);
	pin++; pout++;

	for (i = 1; i < w - 1; i++, pin++, pout++) {
		*pout += bot + KXI(0) + KXI(1) + KXI(2) + KXI(3) + KXI(4) + KXI(5);
	}

	*pout += rb + KXI(0) + KXI(1) + KXI(3) + KXI(4);
}

template<typename T>
void add_conv_k3s1p0(const T* pin, const T* pkernel, T* pout, T pad_value, int w, int h, int pad)
{
	const T* pstart = pin + w + 1;
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

template<typename T>
void add_conv_k1(const T* pin, T kernel, T* pout, int count)
{
	int i;
	for (i = 0; i < count; i++, pin++, pout++) {
		*pout += (*pin * kernel);
	}
}

template<typename T>
static void add_bias(T* ptr, T bias, int count)
{
	for (int i = 0; i < count; i++) {
		ptr[i] += bias;
	}
}

template<typename T>
T get_pad_value(const stInputSource& src);

// 针对 float 的特化
template<>
float get_pad_value<float>(const stInputSource& d) {
	return d.pad_value_f;
}

// 针对 int 的特化
template<>
int get_pad_value<int>(const stInputSource& d) {
	return d.pad_value_i;
}

template<typename T>
T get_pad_value_by_inputc(const UpixLayerInfo* info, int inc)
{
	int c = 0;
	for (auto& from : info->isrcs) {
		if (inc >= c && inc < c + from.channel_count) {
			return get_pad_value<T>(from);
		}
		c += from.channel_count;
	}
	return static_cast<T>(0.0);
}

template<typename T>
static void conv(
	const UpixLayerInfo* info, 
	const TensorShape& inshape, 
	const TensorShape& outshape,	
	const T* pkernel,
	const T* pbias,
	const T* inptr, 
	T* outptr
	)
{
	int i, j, k;
	T* pout;
	const T* pin;

	if (info->ksize == 5) {
		assert(0); // ksize=5时pad太难搞了先不实现了
		assert(info->stride == 1 && info->pad == 2);
		assert(info->groups == outshape.c); // dw
		assert(inshape.w == outshape.w);
		assert(inshape.h == outshape.h);
		assert(inshape.c == outshape.c);

		//for (i = 0; i < info->groups; i++) {
		//	pout = outptr + i * (outshape.whc / info->groups);
		//	for (j = 0; j < outshape.c / info->groups; j++, pout += outshape.wh, pbias++) {
		//		memset(pout, 0, sizeof(T) * outshape.wh);
		//		pin = inptr + i * (inshape.whc / info->groups);
		//		for (k = 0; k < inshape.c / info->groups; k++, pin += inshape.wh, pkernel += 25) {
		//			int inc = i * inshape.c / info->groups + k;
		//			T padv = get_pad_value_by_inputc<T>(info, inc);
		//			add_conv_k5s1p2<T>(pin, pkernel, pout, padv, inshape.w, inshape.h);
		//		}
		//		add_bias<T>(pout, *pbias, outshape.wh);
		//	}
		//}
	}
	else if (info->ksize == 3) {
		if (info->stride == 2 && inshape.w % 2 == 0 && inshape.h % 2 == 0) {
			assert(info->pad == 1);
			for (i = 0; i < info->groups; i++) {
				pout = outptr + i * (outshape.whc / info->groups);
				for (j = 0; j < outshape.c / info->groups; j++, pout += outshape.wh, pbias++) {
					memset(pout, 0, sizeof(T) * outshape.wh);
					pin = inptr + i * (inshape.whc / info->groups);
					for (k = 0; k < inshape.c / info->groups; k++, pin += inshape.wh, pkernel += 9) {
						int inc = (int)(pin - inptr) / inshape.wh;
						T padv = get_pad_value_by_inputc<T>(info, inc);
						add_conv_k3s2p1<T>(pin, pkernel, pout, padv, inshape.w, inshape.h, info->pad);
					}
					add_bias<T>(pout, *pbias, outshape.wh);
				}
			}
		}
		else if (info->stride == 1) {
			if (info->pad == 1) {
				for (i = 0; i < info->groups; i++) {
					pout = outptr + i * (outshape.whc / info->groups);
					for (j = 0; j < outshape.c / info->groups; j++, pout += outshape.wh, pbias++) {
						memset(pout, 0, sizeof(T) * outshape.wh);
						pin = inptr + i * (inshape.whc / info->groups);
						for (k = 0; k < inshape.c / info->groups; k++, pin += inshape.wh, pkernel += 9) {
							int inc = (int)(pin - inptr) / inshape.wh;
							T padv = get_pad_value_by_inputc<T>(info, inc);
							add_conv_k3s1p1<T>(pin, pkernel, pout, padv, inshape.w, inshape.h, info->pad);
						}
						add_bias<T>(pout, *pbias, outshape.wh);
					}
				}
			}
			else if (info->pad == 0) {
				for (i = 0; i < info->groups; i++) {
					pout = outptr + i * (outshape.whc / info->groups);
					for (j = 0; j < outshape.c / info->groups; j++, pout += outshape.wh, pbias++) {
						memset(pout, 0, sizeof(T) * outshape.wh);
						pin = inptr + i * (inshape.whc / info->groups);
						for (k = 0; k < inshape.c / info->groups; k++, pin += inshape.wh, pkernel += 9) {
							int inc = (int)(pin - inptr) / inshape.wh;
							T padv = get_pad_value_by_inputc<T>(info, inc);
							add_conv_k3s1p0<T>(pin, pkernel, pout, padv, inshape.w, inshape.h, info->pad);
						}
						add_bias<T>(pout, *pbias, outshape.wh);
					}
				}
			}
		}
		else {
			assert(0);
		}
	}
	else if (info->ksize == 1) {
		assert(info->pad == 0);
		for (i = 0; i < info->groups; i++) {
			pout = outptr + i * (outshape.whc / info->groups);
			for (j = 0; j < outshape.c / info->groups; j++, pout += outshape.wh, pbias++) {
				memset(pout, 0, sizeof(T) * outshape.wh);
				pin = inptr + i * (inshape.whc / info->groups);
				for (k = 0; k < inshape.c / info->groups; k++, pin += inshape.wh, pkernel++) {
					add_conv_k1<T>(pin, *pkernel, pout, inshape.wh);
				}
				add_bias<T>(pout, *pbias, outshape.wh);
			}
		}
	}
}

void UpixConvLayer::forward_float()
{
	const float* pbias = _bias.fdata.data();
	const float* pkernel = _kernel.fdata.data();
	float* outptr = _output.shape.fptr;
	const float* inptr = _input.shape.fptr;
	
	conv<float>(_info, _input.shape, _output.shape, pkernel, pbias, inptr, outptr);

	// activate
	switch (_info->activate) {
	case ActivateType::Linear: 
		// nothing to do
		break;
	case ActivateType::Leaky:
		for (auto& v : _output.fdata) {
			if (v < 0.0f) {
				v *= 1.0f / 16.0f;
			}
		}
		break;
	case ActivateType::Relu:
		for (auto& v : _output.fdata) {
			if (v < 0.0f) {
				v = 0.0f;
			}
		}
		break;
	case ActivateType::Sigmoid:
		for (auto& v : _output.fdata) {
			v = 1.0f / (1.0f + expf(-v));
		}
		break;
	default: assert(0);
	}
}

#include "../u31/u31_inst.h"
void UpixConvLayer::forward_int()
{
	const int* pbias = _bias.idata.data();
	const int* pkernel = _kernel.idata.data();
	int* outptr = _output.shape.iptr;
	const int* inptr = _input.shape.iptr;

	conv<int>(_info, _input.shape, _output.shape, pkernel, pbias, inptr, outptr);
	for (auto& v : _output.idata) {
		v >>= _info->quantInt.mulShift;
	}

	// activate
	switch (_info->activate) {
	case ActivateType::Linear:
		// nothing to do
		break;
	case ActivateType::Leaky:
		for (auto& v : _output.idata) {
			if (v < 0) {
				v >>= 4;
			}			
		}
		// u31的截断机制
		for (auto& v : _output.idata) {
			if (v > SHRT_MAX) v = SHRT_MAX;
			else if (v < SHRT_MIN) v = SHRT_MIN;
		}

		break;
	case ActivateType::Relu:
		for (auto& v : _output.idata) {
			if (v < 0) {
				v = 0;
			}
		}
		break;
	case ActivateType::Sigmoid:
		for (auto& v : _output.idata) {
			v = u31_sigmoid(v);
		}
		// u31的截断机制
		for (auto& v : _output.idata) {
			if (v > SHRT_MAX) v = SHRT_MAX;
			else if (v < SHRT_MIN) v = SHRT_MIN;
		}
		break;
	default: assert(0);
	}
}
