#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "u31/u31_inst.h"
#include "decode_rects_160x120.h"

#define MAX_COUNT 128
static Target2Type all_rects[MAX_COUNT];
static int rect_count;

extern void whc2cwh(const short* input, short* output, int w, int h, int c);

static int softmax_data[10];
static void softmax_10()
{
	int i, sum = 0;
	for (i = 0; i < 10; i++) {
		softmax_data[i] = u31_exp_m22p10(softmax_data[i]);
		sum += softmax_data[i];
	}
	// 这里必须用浮点, 否则就得直接除才能保证精度
	float inv_sum = 1024.0f / sum;
	for (i = 0; i < 10; i++) {
		softmax_data[i] = (int)(softmax_data[i] * inv_sum);
	}
}

const Target2Type* get_decoded_rects_160x120()
{
	return all_rects;
}

static const float overlap_thres = 0.4f;
static int check_rect_nmls(const Target2Type* r)
{
	int inter_area = 0; // 
	int union_area = 0;

	int right0 = r->x + r->w;
	int bot0 = r->y + r->h;
	int area0 = (int)(((__int64)r->w * r->h) >> 10);
	Target2Type* prect = all_rects;
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


static void decode_rects(const short* c40, const short* c2, int w, int h, int scale)
{
	const short* pscore = c40 + 38;
	const short* p38 = c40;
	const short* p2 = c2;
	int i, j;
	for (i = 0; i < h; i++) {
		for (j = 0; j < w; j++) {
			if (pscore[0] > -1124 || pscore[1] > -1124) {
				Target2Type tar;
				int s0 = u31_sigmoid(pscore[0]);
				int s1 = u31_sigmoid(pscore[1]);
				if (s0 > s1) {
					tar.type = 0;
					tar.score = s0;
				}
				else {
					tar.type = 1;
					tar.score = s1;
				}
				int out[4];

				softmax_data[0] = p2[0];
				softmax_data[1] = p2[1];
				softmax_data[2] = p38[0];
				softmax_data[3] = p38[1];
				softmax_data[4] = p38[2];
				softmax_data[5] = p38[3];
				softmax_data[6] = p38[4];
				softmax_data[7] = p38[5];
				softmax_data[8] = p38[6];
				softmax_data[9] = p38[7];
				softmax_10();
				out[0] = softmax_data[1] + softmax_data[2] * 2 + softmax_data[3] * 3
					+ softmax_data[4] * 4 + softmax_data[5] * 5 + softmax_data[6] * 6
					+ softmax_data[7] * 7 + softmax_data[8] * 8 + softmax_data[9] * 9;

				softmax_data[0] = p38[8];
				softmax_data[1] = p38[9];
				softmax_data[2] = p38[10];
				softmax_data[3] = p38[11];
				softmax_data[4] = p38[12];
				softmax_data[5] = p38[13];
				softmax_data[6] = p38[14];
				softmax_data[7] = p38[15];
				softmax_data[8] = p38[16];
				softmax_data[9] = p38[17];
				softmax_10();
				out[1] = softmax_data[1] + softmax_data[2] * 2 + softmax_data[3] * 3
					+ softmax_data[4] * 4 + softmax_data[5] * 5 + softmax_data[6] * 6
					+ softmax_data[7] * 7 + softmax_data[8] * 8 + softmax_data[9] * 9;

				softmax_data[0] = p38[18];
				softmax_data[1] = p38[19];
				softmax_data[2] = p38[20];
				softmax_data[3] = p38[21];
				softmax_data[4] = p38[22];
				softmax_data[5] = p38[23];
				softmax_data[6] = p38[24];
				softmax_data[7] = p38[25];
				softmax_data[8] = p38[26];
				softmax_data[9] = p38[27];
				softmax_10();
				out[2] = softmax_data[1] + softmax_data[2] * 2 + softmax_data[3] * 3
					+ softmax_data[4] * 4 + softmax_data[5] * 5 + softmax_data[6] * 6
					+ softmax_data[7] * 7 + softmax_data[8] * 8 + softmax_data[9] * 9;

				softmax_data[0] = p38[28];
				softmax_data[1] = p38[29];
				softmax_data[2] = p38[30];
				softmax_data[3] = p38[31];
				softmax_data[4] = p38[32];
				softmax_data[5] = p38[33];
				softmax_data[6] = p38[34];
				softmax_data[7] = p38[35];
				softmax_data[8] = p38[36];
				softmax_data[9] = p38[37];
				softmax_10();
				out[3] = softmax_data[1] + softmax_data[2] * 2 + softmax_data[3] * 3
					+ softmax_data[4] * 4 + softmax_data[5] * 5 + softmax_data[6] * 6
					+ softmax_data[7] * 7 + softmax_data[8] * 8 + softmax_data[9] * 9;

				out[0] = (j << 10) + 512 - out[0];  // w
				out[1] = (i << 10) + 512 - out[1];  // h
				out[2] = (j << 10) + 512 + out[2];
				out[3] = (i << 10) + 512 + out[3];

				tar.w = (out[2] - out[0]) * scale;
				tar.h = (out[3] - out[1]) * scale;
				tar.x = (out[0] + out[2]) / 2 * scale;
				tar.y = (out[1] + out[3]) / 2 * scale;

				if (check_rect_nmls(&tar)) {
					all_rects[rect_count] = tar;
					rect_count++;
				}			
			}
			pscore += 40;
			p38 += 40;
			p2 += 2;
		}
	}
}

int decode_rects_160x120(
	const short* output49, const short* output54,
	int w0, int h0,
	const short* output59, const short* output64,
	int w1, int h1
)
{
	rect_count = 0;

	short* cwh_data = (short*)malloc(sizeof(short) * (w0 * h0 * 42 + w1 * h1 * 42));

	short* cwh_49 = cwh_data;	
	short* cwh_54 = cwh_49 + w0 * h0 * 2;	
	short* cwh_59 = cwh_54 + w0 * h0 * 40;
	short* cwh_64 = cwh_59 + w1 * h1 * 2;
	
	whc2cwh(output49, cwh_49, w0, h0, 2);
	whc2cwh(output54, cwh_54, w0, h0, 40);
	whc2cwh(output59, cwh_59, w1, h1, 2);
	whc2cwh(output64, cwh_64, w1, h1, 40);

	decode_rects(cwh_54, cwh_49, w0, h0, 8);
	decode_rects(cwh_64, cwh_59, w1, h1, 4);
	free(cwh_data);


	return rect_count;
}