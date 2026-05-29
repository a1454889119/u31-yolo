#include <assert.h>
#include "u31/u31_inst.h"
#include "whc2cwh.hpp"
#include "TargetDecoder.h"


typedef struct stTarget_I {
	int type;
	int score;
	int w;
	int h;
	int x; // left
	int y; // top
}Target_I;

static const float overlap_thres = 0.4f;
static int check_rect_nmls(const Target* r, std::vector<Target>& allTargets)
{
	float inter_area = 0; // 
	float union_area = 0;
	
	float right0 = r->x + r->w;
	float bot0 = r->y + r->h;
	float area0 = r->w * r->h;

	for (auto& tar : allTargets) {
		if (tar.type == r->type) {
			float right1 = tar.x + tar.w;
			float left = std::max(r->x, tar.x);
			float right = std::min(right0, right1);
			float inter_w = right - left;
			if (inter_w > 0.0f) {
				float bot1 = tar.y + tar.h;
				float top = std::max(r->y, tar.y);
				float bot = std::min(bot0, bot1);
				float inter_h = bot - top;
				if (inter_h > 0.0f) {
					float area1 = tar.w * tar.h;
					inter_area = inter_w * inter_h;
					union_area = area0 + area1 - inter_area;
					if (inter_area / union_area > overlap_thres) {
						// 到这里已经确定2个box重叠.
						// 下面就是根据得分判断保留哪一个box.
						if (r->score > tar.score) {
							tar = *r; // 替换
						}
						return 0;
					}
				}
			}
		}
	}
	return 1;
}

static int check_rect_nmls_i(const Target_I* r, std::vector<Target_I>& allTargets)
{
	int64_t inter_area = 0; // 
	int64_t union_area = 0;

	int right0 = r->x + r->w;
	int bot0 = r->y + r->h;
	int64_t area0 = (int64_t)r->w * r->h;

	for (auto& tar : allTargets) {
		if (tar.type == r->type) {
			int right1 = tar.x + tar.w;
			int left = std::max(r->x, tar.x);
			int right = std::min(right0, right1);
			int inter_w = right - left;
			if (inter_w > 0.0f) {
				int bot1 = tar.y + tar.h;
				int top = std::max(r->y, tar.y);
				int bot = std::min(bot0, bot1);
				int inter_h = bot - top;
				if (inter_h > 0.0f) {
					int64_t area1 = (int64_t)tar.w * tar.h;
					inter_area = (int64_t)inter_w * inter_h;
					union_area = area0 + area1 - inter_area;
					if ((float)inter_area / union_area > overlap_thres) {
						// 到这里已经确定2个box重叠.
						// 下面就是根据得分判断保留哪一个box.
						if (r->score > tar.score) {
							tar = *r; // 替换
						}
						return 0;
					}
				}
			}
		}
	}
	return 1;
}



static void softmax_float(float* softmax_data, int count)
{
	int i;
	float sum = 0.0f;
	for (i = 0; i < count; i++) {
		softmax_data[i] = expf(softmax_data[i]);
		sum += softmax_data[i];
	}
	// 这里必须用浮点, 否则就得直接除才能保证精度
	float inv_sum = 1.0f / sum;
	for (i = 0; i < count; i++) {
		softmax_data[i] = (softmax_data[i] * inv_sum);
	}
}

static float sigmoid(float v) {
	return 1.0f / (1.0f + expf(-v));
}

static void decode_rects_float(
	const float* c40,
	const float* score_channel,
	float thres,
	int w, int h, int type_count,
	int scale,
	std::vector<Target>& allTargets)
{
	int i, j, k;
	float softmax_data[10];
	for (i = 0; i < h; i++) {
		for (j = 0; j < w; j++) {
			Target tar = { -1, 0.0f };
			for (k = 0; k < type_count; k++) {
				if (score_channel[k] > thres) {
					float s = sigmoid(score_channel[k]);
					if (s > tar.score) {
						tar.score = s;
						tar.type = k;
					}					
				}
			}
			
			if (tar.type >= 0) {
				float out[4];

				softmax_data[0] = c40[0];
				softmax_data[1] = c40[1];
				softmax_data[2] = c40[2];
				softmax_data[3] = c40[3];
				softmax_data[4] = c40[4];
				softmax_data[5] = c40[5];
				softmax_data[6] = c40[6];
				softmax_data[7] = c40[7];
				softmax_data[8] = c40[8];
				softmax_data[9] = c40[9];
				softmax_float(softmax_data, 10);
				out[0] = softmax_data[1] + softmax_data[2] * 2 + softmax_data[3] * 3
					+ softmax_data[4] * 4 + softmax_data[5] * 5 + softmax_data[6] * 6
					+ softmax_data[7] * 7 + softmax_data[8] * 8 + softmax_data[9] * 9;

				softmax_data[0] = c40[10];
				softmax_data[1] = c40[11];
				softmax_data[2] = c40[12];
				softmax_data[3] = c40[13];
				softmax_data[4] = c40[14];
				softmax_data[5] = c40[15];
				softmax_data[6] = c40[16];
				softmax_data[7] = c40[17];
				softmax_data[8] = c40[18];
				softmax_data[9] = c40[19];
				softmax_float(softmax_data, 10);
				out[1] = softmax_data[1] + softmax_data[2] * 2 + softmax_data[3] * 3
					+ softmax_data[4] * 4 + softmax_data[5] * 5 + softmax_data[6] * 6
					+ softmax_data[7] * 7 + softmax_data[8] * 8 + softmax_data[9] * 9;

				softmax_data[0] = c40[20];
				softmax_data[1] = c40[21];
				softmax_data[2] = c40[22];
				softmax_data[3] = c40[23];
				softmax_data[4] = c40[24];
				softmax_data[5] = c40[25];
				softmax_data[6] = c40[26];
				softmax_data[7] = c40[27];
				softmax_data[8] = c40[28];
				softmax_data[9] = c40[29];
				softmax_float(softmax_data, 10);
				out[2] = softmax_data[1] + softmax_data[2] * 2 + softmax_data[3] * 3
					+ softmax_data[4] * 4 + softmax_data[5] * 5 + softmax_data[6] * 6
					+ softmax_data[7] * 7 + softmax_data[8] * 8 + softmax_data[9] * 9;

				softmax_data[0] = c40[30];
				softmax_data[1] = c40[31];
				softmax_data[2] = c40[32];
				softmax_data[3] = c40[33];
				softmax_data[4] = c40[34];
				softmax_data[5] = c40[35];
				softmax_data[6] = c40[36];
				softmax_data[7] = c40[37];
				softmax_data[8] = c40[38];
				softmax_data[9] = c40[39];
				softmax_float(softmax_data, 10);
				out[3] = softmax_data[1] + softmax_data[2] * 2 + softmax_data[3] * 3
					+ softmax_data[4] * 4 + softmax_data[5] * 5 + softmax_data[6] * 6
					+ softmax_data[7] * 7 + softmax_data[8] * 8 + softmax_data[9] * 9;

				out[0] = (float)j - out[0] + 0.5f;  // w
				out[1] = (float)i - out[1] + 0.5f;  // h
				out[2] = (float)j + out[2] + 0.5f;
				out[3] = (float)i + out[3] + 0.5f;

				tar.w = (out[2] - out[0]) * scale;
				tar.h = (out[3] - out[1]) * scale;
				tar.x = (out[0] + out[2]) / 2 * scale;
				tar.y = (out[1] + out[3]) / 2 * scale;

				if (check_rect_nmls(&tar, allTargets)) {
					allTargets.push_back(tar);
				}
			}
			score_channel += type_count;			
			c40 += 40;
		}
	}
}

static void softmax_int(int* softmax_data, int count)
{
	int i, sum = 0;
	for (i = 0; i < count; i++) {
		softmax_data[i] = u31_exp_m22p10(softmax_data[i]);
		sum += softmax_data[i];
	}
	// 这里必须用浮点, 否则就得直接除才能保证精度
	float inv_sum = 1024.0f / sum;
	for (i = 0; i < count; i++) {
		softmax_data[i] = (int)(softmax_data[i] * inv_sum);
	}
}


static void decode_rects_int(
	const int* c40,
	const int* score_channel,
	int thres,
	int w, int h, int type_count,
	int scale,
	std::vector<Target_I>& allTargets)
{
	int i, j, k;
	int softmax_data[10];
	for (i = 0; i < h; i++) {
		for (j = 0; j < w; j++) {
			Target_I tar = { -1, 0 };
			for (k = 0; k < type_count; k++) {
				if (score_channel[k] > thres) {
					int s = u31_sigmoid(score_channel[k]);
					if (s > tar.score) {
						tar.score = s;
						tar.type = k;
					}
				}
			}

			if (tar.type >= 0) {
				int out[4];
				softmax_data[0] = c40[0];
				softmax_data[1] = c40[1];
				softmax_data[2] = c40[2];
				softmax_data[3] = c40[3];
				softmax_data[4] = c40[4];
				softmax_data[5] = c40[5];
				softmax_data[6] = c40[6];
				softmax_data[7] = c40[7];
				softmax_data[8] = c40[8];
				softmax_data[9] = c40[9];
				softmax_int(softmax_data, 10);
				out[0] = softmax_data[1] + softmax_data[2] * 2 + softmax_data[3] * 3
					+ softmax_data[4] * 4 + softmax_data[5] * 5 + softmax_data[6] * 6
					+ softmax_data[7] * 7 + softmax_data[8] * 8 + softmax_data[9] * 9;

				softmax_data[0] = c40[10];
				softmax_data[1] = c40[11];
				softmax_data[2] = c40[12];
				softmax_data[3] = c40[13];
				softmax_data[4] = c40[14];
				softmax_data[5] = c40[15];
				softmax_data[6] = c40[16];
				softmax_data[7] = c40[17];
				softmax_data[8] = c40[18];
				softmax_data[9] = c40[19];
				softmax_int(softmax_data, 10);
				out[1] = softmax_data[1] + softmax_data[2] * 2 + softmax_data[3] * 3
					+ softmax_data[4] * 4 + softmax_data[5] * 5 + softmax_data[6] * 6
					+ softmax_data[7] * 7 + softmax_data[8] * 8 + softmax_data[9] * 9;

				softmax_data[0] = c40[20];
				softmax_data[1] = c40[21];
				softmax_data[2] = c40[22];
				softmax_data[3] = c40[23];
				softmax_data[4] = c40[24];
				softmax_data[5] = c40[25];
				softmax_data[6] = c40[26];
				softmax_data[7] = c40[27];
				softmax_data[8] = c40[28];
				softmax_data[9] = c40[29];
				softmax_int(softmax_data, 10);
				out[2] = softmax_data[1] + softmax_data[2] * 2 + softmax_data[3] * 3
					+ softmax_data[4] * 4 + softmax_data[5] * 5 + softmax_data[6] * 6
					+ softmax_data[7] * 7 + softmax_data[8] * 8 + softmax_data[9] * 9;

				softmax_data[0] = c40[30];
				softmax_data[1] = c40[31];
				softmax_data[2] = c40[32];
				softmax_data[3] = c40[33];
				softmax_data[4] = c40[34];
				softmax_data[5] = c40[35];
				softmax_data[6] = c40[36];
				softmax_data[7] = c40[37];
				softmax_data[8] = c40[38];
				softmax_data[9] = c40[39];
				softmax_int(softmax_data, 10);
				out[3] = softmax_data[1] + softmax_data[2] * 2 + softmax_data[3] * 3
					+ softmax_data[4] * 4 + softmax_data[5] * 5 + softmax_data[6] * 6
					+ softmax_data[7] * 7 + softmax_data[8] * 8 + softmax_data[9] * 9;

				out[0] = ((j << 10) + 512 - out[0]);  // w
				out[1] = ((i << 10) + 512 - out[1]);  // h
				out[2] = ((j << 10) + 512 + out[2]);
				out[3] = ((i << 10) + 512 + out[3]);

				tar.w = ((out[2] - out[0]) * scale) >> 10;
				tar.h = ((out[3] - out[1]) * scale) >> 10;
				tar.x = ((out[0] + out[2]) * scale / 2) >> 10;
				tar.y = ((out[1] + out[3]) * scale / 2) >> 10;
				if (check_rect_nmls_i(&tar, allTargets)) {
					allTargets.push_back(tar);
				}
			}

			c40 += 40;
			score_channel += type_count;
		}
	}
}
#define MAX_TYPE_COUNT 8
#define MAX_W 80
#define MAX_H 80
#define MAX_CWH_DATA_COUNT MAX_W*MAX_H*(40+MAX_TYPE_COUNT)

class TargetDecoderPrivate {
public:
	std::vector<Target> targets;
	std::vector<Target_I> targets_i;
	float thres;
	float cwh_float[MAX_CWH_DATA_COUNT];
	int cwh_int[MAX_CWH_DATA_COUNT];
};

TargetDecoder::TargetDecoder()
{
	d = new TargetDecoderPrivate;
}

TargetDecoder::~TargetDecoder()
{
	delete d;
}

static float inv_sigmoid(float score)
{
	float v = (1.0f - score) / score;
	return (-(logf(v)));

}


void TargetDecoder::reset(float score_thres)
{
	d->targets.clear();
	d->targets_i.clear();
	d->thres = inv_sigmoid(score_thres);
}

int TargetDecoder::decode(const int* scores, const int* c40, int w, int h, int typecount, int scale)
{
	assert(w <= MAX_W && h <= MAX_H);
	int* cwh_scores = d->cwh_int;
	int* cwh_c40 = cwh_scores + w * h * typecount;
	whc2cwh<int>(scores, cwh_scores, w, h, typecount);
	whc2cwh<int>(c40, cwh_c40, w, h, 40);
	decode_rects_int(cwh_c40, cwh_scores, (int)(d->thres * 1024.0f), w, h, typecount, scale, d->targets_i);

	return (int)d->targets_i.size();
}

int TargetDecoder::decode(const float* scores, const float* c40, int w, int h, int typecount, int scale)
{
	assert(w <= MAX_W && h <= MAX_H);
	float* cwh_scores = d->cwh_float;
	float* cwh_c40 = cwh_scores + w * h * typecount;
	whc2cwh<float>(scores, cwh_scores, w, h, typecount);
	whc2cwh<float>(c40, cwh_c40, w, h, 40);
	decode_rects_float(cwh_c40, cwh_scores, d->thres, w, h, typecount, scale, d->targets);

	return (int)d->targets.size();
}

const std::vector<Target>& TargetDecoder::getTargets() const
{
	if (d->targets_i.size() > 0 && d->targets.size() == 0) {
		for (auto& ti : d->targets_i) {
			//printf("0x%08X 0x%08X 0x%08X 0x%08X 0x%08X 0x%08X\n",
			//	ti.type, ti.score, ti.w, ti.h, ti.x, ti.y
			//);
			d->targets.push_back({ ti.type, ti.score / 1024.0f, (float)ti.x, (float)ti.y, (float)ti.w, (float)ti.h });
		}
	}
	return d->targets;
}
