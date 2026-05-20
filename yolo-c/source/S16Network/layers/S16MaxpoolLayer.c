#include <assert.h>
#include "S16MaxpoolLayer.h"

static inline int max2(const s16 *p)
{
	int vmax = p[0];
	if (p[1] > vmax) vmax = p[1];
	return vmax;
}

static inline int max3(const s16 *p)
{
	int vmax = p[0];
	if (p[1] > vmax) vmax = p[1];
	if (p[2] > vmax) vmax = p[2];
	return vmax;
}

static inline int max4(const s16 *p)
{
	int vmax = p[0];
	if (p[1] > vmax) vmax = p[1];
	if (p[2] > vmax) vmax = p[2];
	if (p[3] > vmax) vmax = p[3];
	return vmax;
}

static inline int max5(const s16 *p)
{
	int vmax = p[0];
	if (p[1] > vmax) vmax = p[1];
	if (p[2] > vmax) vmax = p[2];
	if (p[3] > vmax) vmax = p[3];
	if (p[4] > vmax) vmax = p[4];
	return vmax;
}


static inline int max6(const s16 *p)
{
	int vmax = p[0];
	if (p[1] > vmax) vmax = p[1];
	if (p[2] > vmax) vmax = p[2];
	if (p[3] > vmax) vmax = p[3];
	if (p[4] > vmax) vmax = p[4];
	if (p[5] > vmax) vmax = p[5];
	return vmax;
}

static inline int max7(const s16 *p)
{
	int vmax = p[0];
	if (p[1] > vmax) vmax = p[1];
	if (p[2] > vmax) vmax = p[2];
	if (p[3] > vmax) vmax = p[3];
	if (p[4] > vmax) vmax = p[4];
	if (p[5] > vmax) vmax = p[5];
	if (p[6] > vmax) vmax = p[6];
	return vmax;
}

static inline int max8(const s16 *p)
{
	int vmax = p[0];
	if (p[1] > vmax) vmax = p[1];
	if (p[2] > vmax) vmax = p[2];
	if (p[3] > vmax) vmax = p[3];
	if (p[4] > vmax) vmax = p[4];
	if (p[5] > vmax) vmax = p[5];
	if (p[6] > vmax) vmax = p[6];
	if (p[7] > vmax) vmax = p[7];
	return vmax;
}


static void maxpool_row_k8s2p3_ystep5(const s16 *input, s16 *output, int out_w)
{
	int vmax, v, i;
	const s16 *r0 = input;
	const s16 *r1 = r0 + out_w * 2;
	const s16 *r2 = r1 + out_w * 2;
	const s16 *r3 = r2 + out_w * 2;
	const s16 *r4 = r3 + out_w * 2;

#define MAX_5X5 \
	vmax = max5(r0); \
	v = max5(r1); if (v > vmax) vmax = v; \
	v = max5(r2); if (v > vmax) vmax = v; \
	v = max5(r3); if (v > vmax) vmax = v; \
	v = max5(r4); if (v > vmax) vmax = v; \
	*output = vmax; \
	output++;

#define MAX_7X5 \
	vmax = max7(r0); \
	v = max7(r1); if (v > vmax) vmax = v; \
	v = max7(r2); if (v > vmax) vmax = v; \
	v = max7(r3); if (v > vmax) vmax = v; \
	v = max7(r4); if (v > vmax) vmax = v; \
	*output = vmax; \
	output++;

#define MAX_8X5 \
	vmax = max8(r0); \
	v = max8(r1); if (v > vmax) vmax = v; \
	v = max8(r2); if (v > vmax) vmax = v; \
	v = max8(r3); if (v > vmax) vmax = v; \
	v = max8(r4); if (v > vmax) vmax = v; \
	*output = vmax; \
	output++;

	MAX_5X5;
	MAX_7X5;
	r0++, r1++, r2++, r3++, r4++;

	for (i = 2; i < out_w - 2; i++) {
		MAX_8X5;
		r0 += 2, r1 += 2, r2 += 2, r3 += 2, r4 += 2;
	}

	MAX_7X5;
	r0 += 2, r1 += 2, r2 += 2, r3 += 2, r4 += 2;

	MAX_5X5;

#undef MAX_5X5
#undef MAX_7X5
#undef MAX_8X5
}

static void maxpool_row_k8s2p3_ystep7(const s16 *input, s16 *output, int out_w)
{
	int vmax, v, i;
	const s16 *r0 = input;
	const s16 *r1 = r0 + out_w * 2;
	const s16 *r2 = r1 + out_w * 2;
	const s16 *r3 = r2 + out_w * 2;
	const s16 *r4 = r3 + out_w * 2;
	const s16 *r5 = r4 + out_w * 2;
	const s16 *r6 = r5 + out_w * 2;

#define MAX_5X7 \
	vmax = max5(r0); \
	v = max5(r1); if (v > vmax) vmax = v; \
	v = max5(r2); if (v > vmax) vmax = v; \
	v = max5(r3); if (v > vmax) vmax = v; \
	v = max5(r4); if (v > vmax) vmax = v; \
	v = max5(r5); if (v > vmax) vmax = v; \
	v = max5(r6); if (v > vmax) vmax = v; \
	*output = vmax; \
	output++;

#define MAX_7X7 \
	vmax = max7(r0); \
	v = max7(r1); if (v > vmax) vmax = v; \
	v = max7(r2); if (v > vmax) vmax = v; \
	v = max7(r3); if (v > vmax) vmax = v; \
	v = max7(r4); if (v > vmax) vmax = v; \
	v = max7(r5); if (v > vmax) vmax = v; \
	v = max7(r6); if (v > vmax) vmax = v; \
	*output = vmax; \
	output++;

#define MAX_8X7 \
	vmax = max8(r0); \
	v = max8(r1); if (v > vmax) vmax = v; \
	v = max8(r2); if (v > vmax) vmax = v; \
	v = max8(r3); if (v > vmax) vmax = v; \
	v = max8(r4); if (v > vmax) vmax = v; \
	v = max8(r5); if (v > vmax) vmax = v; \
	v = max8(r6); if (v > vmax) vmax = v; \
	*output = vmax; \
	output++;

	MAX_5X7;
	MAX_7X7;
	r0++, r1++, r2++, r3++, r4++, r5++, r6++;

	for (i = 2; i < out_w - 2; i++) {
		MAX_8X7;
		r0 += 2, r1 += 2, r2 += 2, r3 += 2, r4 += 2, r5 += 2, r6 += 2;
	}

	MAX_7X7;
	r0 += 2, r1 += 2, r2 += 2, r3 += 2, r4 += 2, r5 += 2, r6 += 2;

	MAX_5X7;

#undef MAX_5X7
#undef MAX_7X7
#undef MAX_8X7
}

static void maxpool_row_k8s2p3_ystep8(const s16 *input, s16 *output, int out_w)
{
	int vmax, v, i;
	const s16 *r0 = input;
	const s16 *r1 = r0 + out_w * 2;
	const s16 *r2 = r1 + out_w * 2;
	const s16 *r3 = r2 + out_w * 2;
	const s16 *r4 = r3 + out_w * 2;
	const s16 *r5 = r4 + out_w * 2;
	const s16 *r6 = r5 + out_w * 2;
	const s16 *r7 = r6 + out_w * 2;

#define MAX_5X8 \
	vmax = max5(r0); \
	v = max5(r1); if (v > vmax) vmax = v; \
	v = max5(r2); if (v > vmax) vmax = v; \
	v = max5(r3); if (v > vmax) vmax = v; \
	v = max5(r4); if (v > vmax) vmax = v; \
	v = max5(r5); if (v > vmax) vmax = v; \
	v = max5(r6); if (v > vmax) vmax = v; \
	v = max5(r7); if (v > vmax) vmax = v; \
	*output = vmax; \
	output++;

#define MAX_7X8 \
	vmax = max7(r0); \
	v = max7(r1); if (v > vmax) vmax = v; \
	v = max7(r2); if (v > vmax) vmax = v; \
	v = max7(r3); if (v > vmax) vmax = v; \
	v = max7(r4); if (v > vmax) vmax = v; \
	v = max7(r5); if (v > vmax) vmax = v; \
	v = max7(r6); if (v > vmax) vmax = v; \
	v = max7(r7); if (v > vmax) vmax = v; \
	*output = vmax; \
	output++;

#define MAX_8X8 \
	vmax = max8(r0); \
	v = max8(r1); if (v > vmax) vmax = v; \
	v = max8(r2); if (v > vmax) vmax = v; \
	v = max8(r3); if (v > vmax) vmax = v; \
	v = max8(r4); if (v > vmax) vmax = v; \
	v = max8(r5); if (v > vmax) vmax = v; \
	v = max8(r6); if (v > vmax) vmax = v; \
	v = max8(r7); if (v > vmax) vmax = v; \
	*output = vmax; \
	output++;

	MAX_5X8;
	MAX_7X8;
	r0++, r1++, r2++, r3++, r4++, r5++, r6++, r7++;

	for (i = 2; i < out_w - 2; i++) {
		MAX_8X8;
		r0 += 2, r1 += 2, r2 += 2, r3 += 2, r4 += 2, r5 += 2, r6 += 2, r7 += 2;
	}

	MAX_7X8;
	r0 += 2, r1 += 2, r2 += 2, r3 += 2, r4 += 2, r5 += 2, r6 += 2, r7 += 2;

	MAX_5X8;

#undef MAX_5X8
#undef MAX_7X8
#undef MAX_8X8
}

static void maxpool_k8s2p3(const s16 *input, s16 *output, int out_w, int out_h)
{
	int i;
	maxpool_row_k8s2p3_ystep5(input, output, out_w); output += out_w;
	maxpool_row_k8s2p3_ystep7(input, output, out_w); output += out_w; input += out_w * 2;
	for (i = 2; i < out_h - 2; i++) {
		maxpool_row_k8s2p3_ystep8(input, output, out_w); output += out_w; input += out_w * 4;
	}
	maxpool_row_k8s2p3_ystep7(input, output, out_w); output += out_w; input += out_w * 4;
	maxpool_row_k8s2p3_ystep5(input, output, out_w);
}

static void maxpool_row_k4s2p1_ystep3(const s16 *input, s16 *output, int out_w)
{
	int vmax, v, i;
	const s16 *r0 = input;
	const s16 *r1 = r0 + out_w * 2;
	const s16 *r2 = r1 + out_w * 2;

#define MAX_3X3 \
	vmax = max3(r0);\
	v = max3(r1); if (v > vmax) vmax = v;\
	v = max3(r2); if (v > vmax) vmax = v;\
	*output = vmax;\
	output++;

#define MAX_4X3 \
	vmax = max4(r0);\
	v = max4(r1); if (v > vmax) vmax = v;\
	v = max4(r2); if (v > vmax) vmax = v;\
	*output = vmax;\
	output++;

	MAX_3X3;
	r0++, r1++, r2++;

	for (i = 1; i < out_w - 1; i++) {
		MAX_4X3;
		r0 += 2, r1 += 2, r2 += 2;
	}

	MAX_3X3;

#undef MAX_3X3
#undef MAX_4X3
}

static void maxpool_row_k4s2p1_ystep4(const s16 *input, s16 *output, int out_w)
{
	int vmax, v, i;
	const s16 *r0 = input;
	const s16 *r1 = r0 + out_w * 2;
	const s16 *r2 = r1 + out_w * 2;
	const s16 *r3 = r2 + out_w * 2;

#define MAX_3X4 \
	vmax = max3(r0);\
	v = max3(r1); if (v > vmax) vmax = v;\
	v = max3(r2); if (v > vmax) vmax = v;\
	v = max3(r3); if (v > vmax) vmax = v;\
	*output = vmax;\
	output++;

#define MAX_4X4 \
	vmax = max4(r0);\
	v = max4(r1); if (v > vmax) vmax = v;\
	v = max4(r2); if (v > vmax) vmax = v;\
	v = max4(r3); if (v > vmax) vmax = v;\
	*output = vmax;\
	output++;

	MAX_3X4;
	r0++, r1++, r2++; r3++;

	for (i = 1; i < out_w - 1; i++) {
		MAX_4X4;
		r0 += 2, r1 += 2, r2 += 2; r3 += 2;
	}

	MAX_3X4;

#undef MAX_3X4
#undef MAX_4X4
}

static void maxpool_k4s2p1(const s16 *input, s16 *output, int out_w, int out_h)
{
	int i;
	maxpool_row_k4s2p1_ystep3(input, output, out_w); input += out_w * 2; output += out_w;
	for (i = 1; i < out_h - 1; i++) {
		maxpool_row_k4s2p1_ystep4(input, output, out_w); input += out_w * 4; output += out_w;
	}
	maxpool_row_k4s2p1_ystep3(input, output, out_w);
}

static void maxpool_row_k2s2p0(const s16* input, s16* output, int out_w)
{
	int v0, v1, i;
	const s16* r0 = input;
	const s16* r1 = r0 + out_w * 2;
	for (i = 0; i < out_w; i++, r0 += 2, r1 += 2, output++) {
		v0 = max2(r0);
		v1 = max2(r1);
		*output = v0 > v1 ? v0 : v1;
	}
}

static void maxpool_k2s2p0(const s16* input, s16* output, int out_w, int out_h)
{
	int i;
	for (i = 0; i < out_h; i++, input += out_w * 4, output += out_w) {
		maxpool_row_k2s2p0(input, output, out_w);
	}
}

static void maxpool_row_k3s2p1_ystep2(const s16* input, s16* output, int out_w)
{
	int vmax, v, i;
	const s16* r0 = input;
	const s16* r1 = r0 + out_w * 2;

#define MAX_2X2 \
	vmax = max2(r0);\
	v = max2(r1); if (v > vmax) vmax = v;\
	*output = vmax;\
	output++;

#define MAX_2X3 \
	vmax = max3(r0);\
	v = max3(r1); if (v > vmax) vmax = v;\
	*output = vmax;\
	output++;

	MAX_2X2;
	r0++, r1++;
	for (i = 1; i < out_w - 1; i++) {
		MAX_2X3;
		r0 += 2, r1 += 2;
	}
	MAX_2X2;

#undef MAX_3X4
#undef MAX_4X4
}

static void maxpool_k3s2p1(const s16* input, s16* output, int out_w, int out_h)
{
	int i;
	maxpool_row_k3s2p1_ystep2(input, output, out_w); input += out_w * 2; output += out_w;
	for (i = 1; i < out_h - 1; i++) {
		maxpool_row_k4s2p1_ystep4(input, output, out_w); input += out_w * 4; output += out_w;
	}
	
	maxpool_row_k3s2p1_ystep2(input, output, out_w);
}


CAPI void S16_MaxpoolForward (void *userdata, const S16Tensor *input, S16Tensor *output) {
	S16MaxpoolLayer *maxpool = (S16MaxpoolLayer*)userdata;
	const s16 *pin = input->ptr;
	s16 *pout = output->ptr;
	const int input_pixel_count = input->w*input->h;
	const int output_pixel_count = output->w*output->h;
	if (maxpool->size == 8 && maxpool->stride == 2 && maxpool->pad == 3) {
		int i;
		for (i = 0; i < input->c; i++, pin += input_pixel_count, pout += output_pixel_count) {
			maxpool_k8s2p3(pin, pout, output->w, output->h);
		}
	}
	else if (maxpool->size == 4 && maxpool->stride == 2 && maxpool->pad == 1) {
		int i;
		for (i = 0; i < input->c; i++, pin += input_pixel_count, pout += output_pixel_count) {
			maxpool_k4s2p1(pin, pout, output->w, output->h);
		}
	}
	else if (maxpool->size == 2 && maxpool->stride == 2 && maxpool->pad == 0) {
		int i;
		for (i = 0; i < input->c; i++, pin += input_pixel_count, pout += output_pixel_count) {
			maxpool_k2s2p0(pin, pout, output->w, output->h);
		}
	}
	else
	{
		const s16 *pstart;
		s16 vmax;
		int i, j, k, ii;

		for (k = 0; k < input->c; k++, pin += input_pixel_count) {
			for (i = 0; i < input->h; i += maxpool->stride) {
				int ystart = i - maxpool->pad;
				int yend = ystart + maxpool->size - 1;
				int ysteps;
				if (ystart < 0) ystart = 0;
				if (yend >= input->h) yend = input->h - 1;
				ysteps = yend - ystart + 1;

				pstart = pin + ystart * input->w;
				for (j = 0; j < input->w; j += maxpool->stride, pout++) {
					const s16 *ppstart;
					int xstart = j - maxpool->pad;
					int xend = xstart + maxpool->size - 1;
					int xsteps;

					if (xstart < 0) xstart = 0;
					if (xend >= input->w) xend = input->w - 1;
					xsteps = xend - xstart + 1;

					ppstart = pstart + xstart;
					vmax = -32768;
					switch (xsteps) {
					case 0: break;
					case 2:
						for (ii = 0; ii < ysteps; ii++, ppstart += input->w) {
							s16 maxr = max2(ppstart);
							if (maxr > vmax)
								vmax = maxr;
						}
						break;
					case 3:
						for (ii = 0; ii < ysteps; ii++, ppstart += input->w) {
							s16 maxr = max3(ppstart);
							if (maxr > vmax)
								vmax = maxr;
						}
						break;

					case 4:
						for (ii = 0; ii < ysteps; ii++, ppstart += input->w) {
							s16 maxr = max4(ppstart);
							if (maxr > vmax)
								vmax = maxr;
						}
						break;

					case 5:
						for (ii = 0; ii < ysteps; ii++, ppstart += input->w) {
							s16 maxr = max5(ppstart);
							if (maxr > vmax)
								vmax = maxr;
						}
						break;

					case 6:
						for (ii = 0; ii < ysteps; ii++, ppstart += input->w) {
							s16 maxr = max6(ppstart);
							if (maxr > vmax)
								vmax = maxr;
						}
						break;

					case 7:
						for (ii = 0; ii < ysteps; ii++, ppstart += input->w) {
							s16 maxr = max7(ppstart);
							if (maxr > vmax)
								vmax = maxr;
						}
						break;

					case 8:
						for (ii = 0; ii < ysteps; ii++, ppstart += input->w) {
							s16 maxr = max8(ppstart);
							if (maxr > vmax)
								vmax = maxr;
						}
						break;
					default: assert(0); break;
					}

					*pout = vmax;
				}
			}
		}
	}


}
