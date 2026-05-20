#include <stdlib.h>
#include <math.h>

#include "u31/u31_inst.h"
#include "decode_rects.h"

static const int score_thres = 460;

const short anchors_16[6] = { 25, 30, 100, 110, 155, 190 };
const short anchors_32[6] = { 260, 430, 446, 408, 525, 535 };
#define w_16 20
#define h_16 14
#define w_32 10
#define h_32 7
static short c3_0_cwh[w_16 * h_16 * 3];
static short c3_1_cwh[w_16 * h_16 * 3];
static short c12_cwh[w_16 * h_16 * 12];

#define MAX_COUNT 128
static TargetRect all_rects[MAX_COUNT];
static int rect_count;

const TargetRect* get_decoded_rects()
{
	return all_rects;
}

extern void whc2cwh(const short* input, short* output, int w, int h, int c);

int decode_one_rect(
	const short* c3_0, 
	const short* c3_1, 
	const short* c12, 
	const short* anchors,
	int x, int y,
	int down_mul,
	TargetRect* rect)
{
	int s0 = u31_sigmoid(c3_1[0]);
	int s1 = u31_sigmoid(c3_1[1]);
	int s2 = u31_sigmoid(c3_1[2]);
	int smax_1 = max(max(s0, s1), s2);
	if (smax_1 < score_thres)
		return 0;

	const short* p12 = c12;
	const short* panchor = anchors;
	if (smax_1 == s1) {
		p12 += 4;
		panchor += 2;
	}
	else if (smax_1 == s2) {
		p12 += 8;
		panchor += 4;
	}

	s0 = u31_exp_m22p10(c3_0[0]);
	s1 = u31_exp_m22p10(c3_0[1]);
	s2 = u31_exp_m22p10(c3_0[2]);
	int sum = s0 + s1 + s2;
	if (sum == 0) 
		return 0;

	s0 = (int)(((__int64)s0 << 10) / sum);
	s1 = (int)(((__int64)s1 << 10) / sum);
	s2 = (int)(((__int64)s2 << 10) / sum);

	int smax_0 = max(max(s0, s1), s2);
	int type = -1;
	if (smax_0 < score_thres)
		return 0;

	int smax = (smax_0 * smax_1) >> 10;
	if (smax < score_thres)
		return 0;

	if (smax_0 == s0) {
		type = 0;
	}
	else if (smax_0 == s1) {
		type = 1;
	}
	else if (smax_0 == s2) {
		type = 2;
	}

	rect->type = type;
	rect->score = smax;
	rect->w = u31_sigmoid(p12[2]) * 2;
	rect->w = (rect->w * rect->w) >> 10;
	rect->w = rect->w * panchor[0];

	rect->h = u31_sigmoid(p12[3]) * 2;
	rect->h = (rect->h * rect->h) >> 10;
	rect->h = rect->h * panchor[1];

	rect->x = (u31_sigmoid(p12[0]) * 2 - 512 + (x << 10)) * down_mul - (rect->w / 2);
	rect->y = (u31_sigmoid(p12[1]) * 2 - 512 + (y << 10)) * down_mul - (rect->h / 2);
	return 1;
}

static const float overlap_thres = 0.4f;
static int check_rect_nmls(const TargetRect* r)
{
	int inter_area = 0; // 
	int union_area = 0;

	int right0 = r->x + r->w;
	int bot0 = r->y + r->h;
	int area0 = (int)(((__int64)r->w * r->h) >> 10);
	TargetRect* prect = all_rects;
	int i;
	for (i = 0; i < rect_count; i++, prect++) {
		// 类型相同才做nmls
		if (prect->type == r->type) {
			int right1 = prect->x + prect->w;
			int left = max(r->x, prect->x);
			int right = min(right0, right1);
			int inter_w = right - left;
			if (inter_w > 0) {
				int bot1 = prect->y + prect->h;
				int top = max(r->y, prect->y);
				int bot = min(bot0, bot1);
				int inter_h = bot - top;
				if (inter_h > 0) {
					int area1 = (int)(((__int64)prect->w * prect->h) >> 10);
					inter_area = (int)(((__int64)inter_w * inter_h) >> 10);
					union_area = area0 + area1 - inter_area;
					if ((float)inter_area / union_area > overlap_thres) {
						// 到这里已经确定2个box重叠.
						// 下面就是根据得分判断保留哪一个box.
						if (r->score > prect->score) {
							*prect = *r; // 替换
						}
						return 0;
					}
				}
			}
		}
	}
	return 1;
}


static void decode_rects_scale(
	const short* c3_0,
	const short* c3_1, 
	const short* c12, 
	const short* anchors, 
	int w, int h, 
	int down_mul)
{
	TargetRect rect;
	int i, j;
	for (i = 0; i < h; i++) {
		for (j = 0; j < w; j++, c3_0 += 3, c3_1 += 3, c12 += 12) {
			if (decode_one_rect(c3_0, c3_1, c12, anchors, j, i, down_mul, &rect)) {
				if (check_rect_nmls(&rect)) {
					all_rects[rect_count] = rect;
					rect_count++;

					if (rect_count >= MAX_COUNT) {
						return;
					}
				}
			}
		}
	}
}


int decode_rects(
	const short* c3_16_0, const short* c3_16_1, const short* c12_16,
	const short* c3_32_0, const short* c3_32_1, const short* c12_32
) {
	short* p3_0 = c3_0_cwh;
	short* p3_1 = c3_1_cwh;
	short* p12 = c12_cwh;
	rect_count = 0;
	
	whc2cwh(c3_16_0, p3_0, w_16, h_16, 3);
	whc2cwh(c3_16_1, p3_1, w_16, h_16, 3);
	whc2cwh(c12_16, p12, w_16, h_16, 12);
	decode_rects_scale(p3_0, p3_1, p12, anchors_16, w_16, h_16, 16);

	whc2cwh(c3_32_0, p3_0, w_32, h_32, 3);
	whc2cwh(c3_32_1, p3_1, w_32, h_32, 3);
	whc2cwh(c12_32, p12, w_32, h_32, 12);
	decode_rects_scale(p3_0, p3_1, p12, anchors_32, w_32, h_32, 32);

	return rect_count;
}
