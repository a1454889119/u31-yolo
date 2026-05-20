#ifndef __KALMAN_BOX_TRACKER_H__
#define __KALMAN_BOX_TRACKER_H__

#include <opencv2/opencv.hpp>
#include "KalmanTracker.h"

class KalmanBoxTracker {
public:
    static int g_global_id;

    KalmanBoxTracker(const BBox& bbox, int img_w = TrackerConfig::IMG_W, int img_h = TrackerConfig::IMG_H) {
        float cx = (bbox.x1 + bbox.x2) / 2;
        float cy = (bbox.y1 + bbox.y2) / 2;
        float bw = bbox.x2 - bbox.x1;
        float bh = bbox.y2 - bbox.y1;
        float s = bw * bh;
        float r = bw / bh;

        // 7状态: [cx, cy, s, r, vx, vy, vs]
        // 4测量: [cx, cy, s, r]
        state = cv::Mat::zeros(7, 1, CV_32F);
        state.at<float>(0) = cx;
        state.at<float>(1) = cy;
        state.at<float>(2) = s;
        state.at<float>(3) = r;
        state.at<float>(4) = 0.0f; // vx
        state.at<float>(5) = 0.0f; // vy
        state.at<float>(6) = 0.0f; // vs

        // 状态协方差矩阵 P
        P = cv::Mat::eye(7, 7, CV_32F) * 1e-1f;

        // 测量矩阵 H (4x7)
        H = cv::Mat::zeros(4, 7, CV_32F);
        H.at<float>(0, 0) = 1.0f;
        H.at<float>(1, 1) = 1.0f;
        H.at<float>(2, 2) = 1.0f;
        H.at<float>(3, 3) = 1.0f;

        // 过程噪声 Q (Heavy Kalman策略: vx/vy noise=0.1)
        Q = cv::Mat::zeros(7, 7, CV_32F);
        Q.at<float>(0, 0) = 0.5f;
        Q.at<float>(1, 1) = 0.5f;
        Q.at<float>(2, 2) = 0.01f;
        Q.at<float>(3, 3) = 0.001f;
        Q.at<float>(4, 4) = 0.1f;
        Q.at<float>(5, 5) = 0.1f;
        Q.at<float>(6, 6) = 0.1f;

        // 测量噪声 R
        R = cv::Mat::zeros(4, 4, CV_32F);
        R.at<float>(0, 0) = 2.0f;
        R.at<float>(1, 1) = 2.0f;
        R.at<float>(2, 2) = 10.0f;
        R.at<float>(3, 3) = 1.0f;

        img_width = img_w;
        img_height = img_h;

        id = g_global_id++;
        hits = 0;
        consecutive_hits = 0;
        time_since_update = 0;
        state_track = TENTATIVE;
    }

    void setImageSize(int w, int h) {
        img_width = w;
        img_height = h;
    }

    // 预测
    BBox predict() {
        // 状态转移矩阵 F (匀速模型)
        cv::Mat F = cv::Mat::eye(7, 7, CV_32F);
        F.at<float>(0, 4) = 1.0f;
        F.at<float>(1, 5) = 1.0f;
        F.at<float>(2, 6) = 1.0f;

        // 速度衰减: 丢失帧越多，速度逐渐衰减到接近0
        // 这样当目标静止或慢速运动时，预测不会大幅漂移
        if (time_since_update > 0) {
            float decay = std::pow(TrackerConfig::VEL_DECAY_ON_MISS, (float)time_since_update);
            state.at<float>(4) *= decay;
            state.at<float>(5) *= decay;
            state.at<float>(6) *= decay;
        }

        // x = F * x
        state = F * state;

        // P = F * P * F' + Q
        P = F * P * F.t() + Q;

        // 面积/宽高比保护
        float s = state.at<float>(2);
        s = std::max(TrackerConfig::MIN_AREA, s);
        state.at<float>(2) = s;

        float r = state.at<float>(3);
        r = std::max(TrackerConfig::MIN_ASPECT, std::min(TrackerConfig::MAX_ASPECT, r));
        state.at<float>(3) = r;

        time_since_update++;
        if (time_since_update > 1) {
            consecutive_hits = 0;
        }

        // FIX 1: 将预测框严格截断到图像边界内，防止框飞出图像
        BBox box = _stateToBox();
        box.x1 = std::max(0.0f, std::min(box.x1, (float)img_width));
        box.y1 = std::max(0.0f, std::min(box.y1, (float)img_height));
        box.x2 = std::max(0.0f, std::min(box.x2, (float)img_width));
        box.y2 = std::max(0.0f, std::min(box.y2, (float)img_height));

        //// 如果截断后面积太小，恢复到上一帧状态
        //if (box.w() < 2.0f || box.h() < 2.0f) {
        //    return _stateToBox(); // 回退到未截断版本（保留合理性保护）
        //}

        //// 同时修正状态中的cx/cy，避免状态与返回的box不一致
        //state.at<float>(0) = box.cx();
        //state.at<float>(1) = box.cy();

        return box;
    }

    // 更新
    void update(const BBox& bbox) {
        float cx = (bbox.x1 + bbox.x2) / 2;
        float cy = (bbox.y1 + bbox.y2) / 2;
        float bw = bbox.x2 - bbox.x1;
        float bh = bbox.y2 - bbox.y1;
        float s = bw * bh;
        float r = bw / bh;

        // FIX: 测量值也应该截断到图像边界内（与 predict() 保持一致）
        // 这样创新向量 y = 测量 - 预测 的参考基准相同
        // 避免目标在边界附近时，滤波器错误地把"边界截断差"当成速度信号
        float meas_half_h = std::sqrt(s / r) * 0.5f;
        cx = std::max(meas_half_h, std::min(cx, (float)img_width - meas_half_h));
        cy = std::max(meas_half_h, std::min(cy, (float)img_height - meas_half_h));

        // FIX 2: 方向反转检测
        // 如果测量值与预测值偏离过大(超过阈值)，说明方向发生了突变
        // 此时不应该让卡尔曼增益过度信任观测值，而应该重置速度状态
        float pred_cx = state.at<float>(0);
        float pred_cy = state.at<float>(1);
        float meas_dx = cx - pred_cx;
        float meas_dy = cy - pred_cy;
        float meas_dist = std::sqrt(meas_dx * meas_dx + meas_dy * meas_dy);

        if (meas_dist > TrackerConfig::REVERSE_DETECT_THRESHOLD) {
            // 方向反转：计算测量值相对于当前状态的"真实"速度
            // 用一个小缩放因子初始化新速度，避免旧速度惯性太大
            float new_vx = meas_dx * TrackerConfig::REVERSE_VEL_SCALE;
            float new_vy = meas_dy * TrackerConfig::REVERSE_VEL_SCALE;

            // 重置状态中的速度为小值，让滤波器快速收敛到新方向
            // 同时增大P矩阵中对应速度的协方差，让滤波器更快适应新速度
            state.at<float>(4) = new_vx;
            state.at<float>(5) = new_vy;
            state.at<float>(6) = 0.0f;

            // 增大P矩阵中对应速度的协方差，让滤波器对新测量更敏感
            P.at<float>(4, 4) = 10.0f;
            P.at<float>(5, 5) = 10.0f;
            P.at<float>(6, 6) = 1.0f;
        }

        // 测量向量 z = [cx, cy, s, r]'
        cv::Mat z = (cv::Mat_<float>(4, 1) << cx, cy, s, r);

        // 预测测量: z_hat = H * x
        cv::Mat z_hat = H * state;

        // 创新: y = z - z_hat
        cv::Mat y = z - z_hat;

        // 创新协方差: S = H * P * H' + R
        cv::Mat S = H * P * H.t() + R;

        // 卡尔曼增益: K = P * H' * S^(-1)
        cv::Mat K = P * H.t() * S.inv();

        // 更新状态: x = x + K * y
        state = state + K * y;

        // 更新协方差: P = (I - K * H) * P
        P = (cv::Mat::eye(7, 7, CV_32F) - K * H) * P;

        // FIX 3: 更新后再次截断到图像边界，防止极端测量值导致状态越界
        //BBox box = _stateToBox();
        //float half_w = box.w() * 0.5f;
        //float half_h = box.h() * 0.5f;
        //float clamped_cx = std::max(half_w, std::min(pred_cx, (float)img_width - half_w));
        //float clamped_cy = std::max(half_h, std::min(pred_cy, (float)img_height - half_h));
        //if (std::abs(state.at<float>(0) - clamped_cx) > half_w * 0.5f ||
        //    std::abs(state.at<float>(1) - clamped_cy) > half_h * 0.5f) {
        //    state.at<float>(0) = clamped_cx;
        //    state.at<float>(1) = clamped_cy;
        //}

        hits++;
        time_since_update = 0;
        consecutive_hits++;

        if (state_track == TENTATIVE && consecutive_hits >= TrackerConfig::MIN_HITS) {
            state_track = CONFIRMED;
        }
    }

    BBox getState() const {
        return _stateToBox();
    }

    void getWHR(float& w, float& h) const {
        float s = state.at<float>(2);
        float r = state.at<float>(3);
        s = std::max(TrackerConfig::MIN_AREA, s);
        r = std::max(TrackerConfig::MIN_ASPECT, std::min(TrackerConfig::MAX_ASPECT, r));
        h = std::sqrt(s / r);
        w = r * h;
    }

    void getVelocity(float& vx, float& vy) const {
        vx = state.at<float>(4);
        vy = state.at<float>(5);
    }

    float getSpeed() const {
        float vx = state.at<float>(4);
        float vy = state.at<float>(5);
        return std::sqrt(vx * vx + vy * vy);
    }

    int  getID() const { return id; }
    int  getTimeSinceUpdate() const { return time_since_update; }
    bool isConfirmed() const { return state_track == CONFIRMED; }
    bool isTentative() const { return state_track == TENTATIVE; }

    int getImageWidth()  const { return img_width; }
    int getImageHeight() const { return img_height; }

private:
    BBox _stateToBox() const {
        float s = state.at<float>(2);
        float r = state.at<float>(3);
        s = std::max(TrackerConfig::MIN_AREA, s);
        r = std::max(TrackerConfig::MIN_ASPECT, std::min(TrackerConfig::MAX_ASPECT, r));
        float h = std::sqrt(s / r);
        float w = r * h;
        float cx = state.at<float>(0);
        float cy = state.at<float>(1);
        return { cx - w * 0.5f, cy - h * 0.5f, cx + w * 0.5f, cy + h * 0.5f };
    }

    cv::Mat state;  // 7x1
    cv::Mat P;      // 7x7
    cv::Mat H;      // 4x7
    cv::Mat Q;      // 7x7
    cv::Mat R;      // 4x4

    int img_width;
    int img_height;

public:
    int id;
    int hits;
    int consecutive_hits;
    int time_since_update;
    int state_track;
};



#endif // !__KALMAN_BOX_TRACKER_H__
