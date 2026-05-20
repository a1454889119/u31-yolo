#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include "u31_sqrtf.h"
#include "KalmanTracker.h"

//namespace TrackerConfig {
//	static constexpr float CONF_THRESHOLD = 0.25f;
//	static constexpr int   MIN_HITS = 2;
//	static constexpr int   UNLOCK_AFTER_N_FRAMES = 10;
//	static constexpr int   AUTO_RELOCK_AGE_THRESHOLD = 20;
//	static constexpr float GATE_BASE = 50.0f;
//	static constexpr float GATE_SCALE_W = 2.0f;
//	static constexpr float GATE_SCALE_V = 0.5f;
//	static constexpr float GATE_SHRINK_STEP = 0.3f;
//	static constexpr float GATE_SHRINK_MIN = 30.0f;
//	static constexpr float MAX_AREA_RATIO = 3.0f;
//	static constexpr float MIN_AREA = 300.0f;
//	static constexpr float MIN_ASPECT = 0.3f;
//	static constexpr float MAX_ASPECT = 3.3f;
//
//	// 图像尺寸 (用于边界截断)
//	static constexpr int IMG_W = 320;
//	static constexpr int IMG_H = 160;
//
//	// 速度衰减与反转检测参数
//	static constexpr float VEL_DECAY_ON_MISS = 0.80f;    // 丢失帧时速度衰减系数
//	static constexpr float REVERSE_DETECT_THRESHOLD = 50.0f; // 检测方向反转的距离阈值(像素)
//	static constexpr float REVERSE_VEL_SCALE = 0.05f;    // 反转后速度重置时的缩放因子
//};

struct stKalmanTracker {
	float state[7];
	float P[7 * 7];
	float H[7 * 4];
	float Q[7 * 7];
	float R[4 * 4];
	int imgw;
	int imgh;
	int time_since_update;

	/*
		0 - - - 7 - -
		- 1 - - - 8 -
		- - 2 - - - 9
		- - - 3 - - -    
		a - - - 4 - -
		- b	- - - 5 -
		- - c - - - 6
		a:10
		b:11
		c:12
	*/
	float pmat[13]; // 对角7 + 6
	
};

static const float default_pmat[13] = {
	0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f,
	0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
};

static const float qmat[7] = {
	0.5f,	0.5f,	0.01f,	0.001f,
	0.1f,	0.1f,	0.1f,
};

static const float rmat[4] = {
	2.0f,	2.0f,	10.0f,	1.0f
};

// 状态协方差矩阵 P
static const float default_P[7*7] = {
	0.1f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
	0.0f, 0.1f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
	0.0f, 0.0f, 0.1f, 0.0f, 0.0f, 0.0f, 0.0f,
	0.0f, 0.0f, 0.0f, 0.1f, 0.0f, 0.0f, 0.0f,
	0.0f, 0.0f, 0.0f, 0.0f, 0.1f, 0.0f, 0.0f,
	0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.1f, 0.0f,
	0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.1f,
};



// 测量矩阵 H (4x7)
static const float default_H[7 * 4] = {
	1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
	0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
	0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
	0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
};

// 过程噪声 Q (Heavy Kalman策略: vx/vy noise=0.1)
static const float default_Q[7 * 7] = {
	0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
	0.0f, 0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
	0.0f, 0.0f, 0.01f, 0.0f, 0.0f, 0.0f, 0.0f,
	0.0f, 0.0f, 0.0f, 0.001f, 0.0f, 0.0f, 0.0f,
	0.0f, 0.0f, 0.0f, 0.0f, 0.1f, 0.0f, 0.0f,
	0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.1f, 0.0f,
	0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.1f,
};

// 测量噪声 R
static const float default_R[4 * 4] = {
	2.0f, 0.0f, 0.0f, 0.0f,
	0.0f, 2.0f, 0.0f, 0.0f,
	0.0f, 0.0f, 10.0f, 0.0f,
	0.0f, 0.0f, 0.0f, 1.0f,
};

KalmanTracker* KalmanTracker_Create(int imgw, int imgh)
{
	KalmanTracker* kt = (KalmanTracker*)malloc(sizeof(KalmanTracker));
	if (kt) {
		kt->imgw = imgw;
		kt->imgh = imgh;
		kt->time_since_update = 0;
	}
	return kt;
}

void KalmanTracker_Destroy(KalmanTracker* kt)
{
	if (kt) free(kt);
}

TrackState KalmanTracker_Reset(KalmanTracker* kt, const BBox* box)
{
	TrackState state = TENTATIVE;
	float cx = (box->x1 + box->x2) / 2;
	float cy = (box->y1 + box->y2) / 2;
	float bw = box->x2 - box->x1;
	float bh = box->y2 - box->y1;
	float s = bw * bh;
	float r = bw / bh;

	kt->state[0] = cx;
	kt->state[1] = cy;
	kt->state[2] = s;
	kt->state[3] = r;
	kt->state[4] = 0.0f;
	kt->state[5] = 0.0f;
	kt->state[6] = 0.0f;

	memcpy(kt->P, default_P, sizeof(default_P));
	memcpy(kt->Q, default_Q, sizeof(default_Q));
	memcpy(kt->R, default_R, sizeof(default_R));
	memcpy(kt->H, default_H, sizeof(default_H));
	kt->time_since_update = 0;

	memcpy(kt->pmat, default_pmat, sizeof(default_pmat));
	return state;
}

TrackState KalmanTracker_Predict(KalmanTracker* kt, BBox* box)
{
	TrackState state = TENTATIVE;
	// 状态转移矩阵 F (匀速模型)
	cv::Mat F = cv::Mat::eye(7, 7, CV_32F);
	F.at<float>(0, 4) = 1.0f;
	F.at<float>(1, 5) = 1.0f;
	F.at<float>(2, 6) = 1.0f;

	// 速度衰减: 丢失帧越多，速度逐渐衰减到接近0
	// 这样当目标静止或慢速运动时，预测不会大幅漂移
	int t = kt->time_since_update;
	kt->time_since_update++;
	float decay = 1.0f;
	while (t > 0) {
		decay *= 0.8f;
		t--;
	}
	kt->state[4] *= decay;
	kt->state[5] *= decay;
	kt->state[6] *= decay;

	cv::Mat x(7, 1, CV_32FC1, (void*)kt->state);
	// x = F * x
	kt->state[0] += kt->state[4];
	kt->state[1] += kt->state[5];
	kt->state[2] += kt->state[6];


	// P = F * P * F' + Q
	// opencv部分
	cv::Mat P(7, 7, CV_32FC1, kt->P);
	cv::Mat Q(7, 7, CV_32FC1, kt->Q);
	P = F * P * F.t() + Q;
	// pmat部分
	/* 原始pmat:
		A - - - K - -
		- B - - - L -
		- - C - - - M
		- - - D - - -
		H - - - E - -
		- I	- - - F -
		- - J - - - G
		a:10
		b:11
		c:12

		结果:
		[A + E + H + K, 0, 0, 0, E + K, 0, 0],
		[0, B + F + I + L, 0, 0, 0, F + L, 0],
		[0, 0, C + G + J + M, 0, 0, 0, G + M],
		[0, 0, 0, D, 0, 0, 0],
		[E + H, 0, 0, 0, E, 0, 0],
		[0, F + I, 0, 0, 0, F, 0],
		[0, 0, G + J, 0, 0, 0, G]]

	*/
	kt->pmat[0] = kt->pmat[0] + kt->pmat[4] + kt->pmat[7] + kt->pmat[10];
	kt->pmat[1] = kt->pmat[1] + kt->pmat[5] + kt->pmat[8] + kt->pmat[11];
	kt->pmat[2] = kt->pmat[2] + kt->pmat[6] + kt->pmat[9] + kt->pmat[12];
	kt->pmat[7] = kt->pmat[4] + kt->pmat[7];
	kt->pmat[8] = kt->pmat[5] + kt->pmat[8];
	kt->pmat[9] = kt->pmat[6] + kt->pmat[9];
	kt->pmat[10] = kt->pmat[4] + kt->pmat[10];
	kt->pmat[11] = kt->pmat[5] + kt->pmat[11];
	kt->pmat[12] = kt->pmat[6] + kt->pmat[12];

	kt->pmat[0] += qmat[0];
	kt->pmat[1] += qmat[1];
	kt->pmat[2] += qmat[2];
	kt->pmat[3] += qmat[3];
	kt->pmat[4] += qmat[4];
	kt->pmat[5] += qmat[5];
	kt->pmat[6] += qmat[6];
	// 验证
	cv::Mat tmp = cv::Mat::zeros(7, 7, CV_32FC1);
	tmp.at<float>(0, 0) = kt->pmat[0];
	tmp.at<float>(1, 1) = kt->pmat[1];
	tmp.at<float>(2, 2) = kt->pmat[2];
	tmp.at<float>(3, 3) = kt->pmat[3];
	tmp.at<float>(4, 4) = kt->pmat[4];
	tmp.at<float>(5, 5) = kt->pmat[5];
	tmp.at<float>(6, 6) = kt->pmat[6];
	tmp.at<float>(0, 4) = kt->pmat[7];
	tmp.at<float>(1, 5) = kt->pmat[8];
	tmp.at<float>(2, 6) = kt->pmat[9];
	tmp.at<float>(4, 0) = kt->pmat[10];
	tmp.at<float>(5, 1) = kt->pmat[11];
	tmp.at<float>(6, 2) = kt->pmat[12];


	// 面积/宽高比保护
	kt->state[2] = std::max(300.0f, kt->state[2]);
	kt->state[3] = std::min(3.3f, std::max(0.3f, kt->state[3]));

	// _stateToBox
	float h = sqrtf(kt->state[2] / kt->state[3]);
	float w = h * kt->state[3];
	float cx = kt->state[0];
	float cy = kt->state[1];

	box->x1 = cx - w * 0.5f;
	box->y1 = cy - h * 0.5f;
	box->x2 = cx + w * 0.5f;
	box->y2 = cy + h * 0.5f;
	// 将预测框严格截断到图像边界内，防止框飞出图像
	box->x1 = std::max(0.0f, std::min(box->x1, (float)kt->imgw));
	box->y1 = std::max(0.0f, std::min(box->y1, (float)kt->imgh));
	box->x2 = std::max(0.0f, std::min(box->x2, (float)kt->imgw));
	box->y2 = std::max(0.0f, std::min(box->y2, (float)kt->imgh));

	return state;
}

TrackState KalmanTracker_Update(KalmanTracker* kt, const BBox* box)
{
	TrackState state = TENTATIVE;

	float cx = (box->x1 + box->x2) / 2;
	float cy = (box->y1 + box->y2) / 2;
	float bw = box->x2 - box->x1;
	float bh = box->y2 - box->y1;
	float s = bw * bh;
	float r = bw / bh;

	// FIX: 测量值也应该截断到图像边界内（与 predict() 保持一致）
	// 这样创新向量 y = 测量 - 预测 的参考基准相同
	// 避免目标在边界附近时，滤波器错误地把"边界截断差"当成速度信号
	float meas_half_h = bh * 0.5f;
	cx = std::max(meas_half_h, std::min(cx, (float)kt->imgw - meas_half_h));
	cy = std::max(meas_half_h, std::min(cy, (float)kt->imgh - meas_half_h));

	// FIX 2: 方向反转检测
	// 如果测量值与预测值偏离过大(超过阈值)，说明方向发生了突变
	// 此时不应该让卡尔曼增益过度信任观测值，而应该重置速度状态
	float pred_cx = kt->state[0];
	float pred_cy = kt->state[1];
	float meas_dx = cx - pred_cx;
	float meas_dy = cy - pred_cy;
	float meas_dist = sqrtf(meas_dx * meas_dx + meas_dy * meas_dy);

	if (meas_dist > TrackerConfig::REVERSE_DETECT_THRESHOLD) {
		// 方向反转：计算测量值相对于当前状态的"真实"速度
		// 用一个小缩放因子初始化新速度，避免旧速度惯性太大
		float new_vx = meas_dx * TrackerConfig::REVERSE_VEL_SCALE;
		float new_vy = meas_dy * TrackerConfig::REVERSE_VEL_SCALE;

		// 重置状态中的速度为小值，让滤波器快速收敛到新方向
		// 同时增大P矩阵中对应速度的协方差，让滤波器更快适应新速度
		kt->state[4] = new_vx;
		kt->state[5] = new_vy;
		kt->state[6] = 0.0f;

		// 增大P矩阵中对应速度的协方差，让滤波器对新测量更敏感
		kt->P[4 * 7 + 4] = 10.0f;
		kt->P[5 * 7 + 5] = 10.0f;
		kt->P[6 * 7 + 6] = 1.0f;
		kt->pmat[4] = 10.0f;
		kt->pmat[5] = 10.0f;
		kt->pmat[6] = 1.0f;
	}


	// 测量向量 z = [cx, cy, s, r]'

	cv::Mat H(4, 7, CV_32F, kt->H);
	cv::Mat P(7, 7, CV_32F, kt->P);
	cv::Mat R(4, 4, CV_32F, kt->R);


	// 创新协方差: S = H * P * H' + R
	cv::Mat S = H * P * H.t() + R;
	float smat[4] = {
		kt->pmat[0] + rmat[0],
		kt->pmat[1] + rmat[1],
		kt->pmat[2] + rmat[2],
		kt->pmat[3] + rmat[3]
	};
	// 卡尔曼增益: K = P * H' * S^(-1)
	cv::Mat sinv = S.inv();
	cv::Mat pht = P * H.t();
	cv::Mat K = pht * sinv;
	float kmat[7] = {
		kt->pmat[0],
		kt->pmat[1],
		kt->pmat[2],
		kt->pmat[3],
		kt->pmat[7],
		kt->pmat[8],
		kt->pmat[9],
	};
	if (fabsf(smat[0]) > FLT_EPSILON) {
		kmat[0] /= smat[0];
		kmat[4] /= smat[0];
	}
	else {
		kmat[0] = 0.0f;
		kmat[4] = 0.0f;
	}
	if (fabsf(smat[1]) > FLT_EPSILON) {
		kmat[1] /= smat[1];
		kmat[5] /= smat[1];
	}
	else {
		kmat[1] = 0.0f;
		kmat[5] = 0.0f;
	}
	if (fabsf(smat[2]) > FLT_EPSILON) {
		kmat[2] /= smat[2];
		kmat[6] /= smat[2];
	}
	else {
		kmat[2] = 0.0f;
		kmat[6] = 0.0f;
	}
	if (fabsf(smat[3]) > FLT_EPSILON) {
		kmat[3] /= smat[3];
	}
	else {
		kmat[3] = 0.0f;
	}
	float IKH[7] = {
		1.0f - kmat[0],
		1.0f - kmat[1],
		1.0f - kmat[2],
		1.0f - kmat[3],
		-kmat[4],
		-kmat[5],
		-kmat[6]
	};

	cv::Mat kh = K * H;
	cv::Mat pp = cv::Mat::eye(7, 7, CV_32F) - kh;
	// 更新协方差: P = (I - K * H) * P
	P = (cv::Mat::eye(7, 7, CV_32F) - K * H) * P;

	kt->pmat[7] = kt->pmat[7] + (kt->pmat[0] * IKH[4]);
	kt->pmat[8] = kt->pmat[8] + (kt->pmat[1] * IKH[5]);
	kt->pmat[9] = kt->pmat[9] + (kt->pmat[2] * IKH[6]);
	kt->pmat[0] = kt->pmat[0] * IKH[0];
	kt->pmat[1] = kt->pmat[1] * IKH[1];
	kt->pmat[2] = kt->pmat[2] * IKH[2];
	kt->pmat[3] = kt->pmat[3] * IKH[3];
	kt->pmat[4] = kt->pmat[4] + (kt->pmat[10] * IKH[4]);
	kt->pmat[5] = kt->pmat[5] + (kt->pmat[11] * IKH[5]);
	kt->pmat[6] = kt->pmat[6] + (kt->pmat[12] * IKH[6]);
	kt->pmat[10] = kt->pmat[10] * IKH[0];
	kt->pmat[11] = kt->pmat[11] * IKH[1];
	kt->pmat[12] = kt->pmat[12] * IKH[2];

	// 预测测量: z_hat = H * x
	cv::Mat z = (cv::Mat_<float>(4, 1) << cx, cy, s, r);
	cv::Mat X(7, 1, CV_32F, kt->state);
	cv::Mat z_hat = H * X;

	// 创新: y = z - z_hat
	cv::Mat y = z - z_hat;
	cv::Mat ky = K * y;
	// 更新状态: x = x + K * y

	float ymat[4] = {
		cx - kt->state[0],
		cy - kt->state[1],
		s - kt->state[2],
		r - kt->state[3]
	};
	float kymat[7] = {
		kmat[0] * ymat[0],
		kmat[1] * ymat[1],
		kmat[2] * ymat[2],
		kmat[3] * ymat[3],
		kmat[4] * ymat[0],
		kmat[5] * ymat[1],
		kmat[6] * ymat[2],
	};
	X = X + K * y;
	kt->time_since_update = 0;
	return state;
}
