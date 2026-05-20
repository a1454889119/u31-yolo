#ifndef __KALMAN_TRACKER_H__
#define __KALMAN_TRACKER_H__

typedef struct stKalmanTracker KalmanTracker;

// ============================================================================
// 跟踪器状态
// ============================================================================
typedef enum enTrackState {
    TENTATIVE = 0,
    CONFIRMED = 1,
    LOST = 2,
    REMOVED = 3
}TrackState;

// ============================================================================
// 跟踪器配置参数
// ============================================================================
struct TrackerConfig {
    static constexpr float CONF_THRESHOLD = 0.25f;
    static constexpr int   MIN_HITS = 2;
    static constexpr int   UNLOCK_AFTER_N_FRAMES = 10;
    static constexpr int   AUTO_RELOCK_AGE_THRESHOLD = 20;
    static constexpr float GATE_BASE = 50.0f;
    static constexpr float GATE_SCALE_W = 2.0f;
    static constexpr float GATE_SCALE_V = 0.5f;
    static constexpr float GATE_SHRINK_STEP = 0.3f;
    static constexpr float GATE_SHRINK_MIN = 30.0f;
    static constexpr float MAX_AREA_RATIO = 3.0f;
    static constexpr float MIN_AREA = 300.0f;
    static constexpr float MIN_ASPECT = 0.3f;
    static constexpr float MAX_ASPECT = 3.3f;

    // 图像尺寸 (用于边界截断)
    static constexpr int IMG_W = 320;
    static constexpr int IMG_H = 160;

    // 速度衰减与反转检测参数
    static constexpr float VEL_DECAY_ON_MISS = 0.80f;    // 丢失帧时速度衰减系数
    static constexpr float REVERSE_DETECT_THRESHOLD = 50.0f; // 检测方向反转的距离阈值(像素)
    static constexpr float REVERSE_VEL_SCALE = 0.05f;    // 反转后速度重置时的缩放因子
};

typedef struct stBBox {
    float x1, y1, x2, y2;
    float score;
    int class_id;
}BBox;

#ifdef __cplusplus
extern "C" {
#endif

KalmanTracker* KalmanTracker_Create(int imgw, int imgh);

void KalmanTracker_Destroy(KalmanTracker* kt);

TrackState KalmanTracker_Reset(KalmanTracker* kt, const BBox* box);

TrackState KalmanTracker_Predict(KalmanTracker* kt, BBox* box);

TrackState KalmanTracker_Update(KalmanTracker* kt, const BBox* box);

#ifdef __cplusplus
}
#endif

#endif