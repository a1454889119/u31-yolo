#include <filesystem>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <UpixNetwork/Layers/UpixBaseLayer.h>
#include <UpixNetwork/UpixNetwork.h>
#include <algorithm>
#include <vector>
#include <string>
#include <cmath>


#define CV_VERSION_ID CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)
#ifdef _DEBUG
#define cvLIB(name) "opencv_" name CV_VERSION_ID "d"
#else
#define cvLIB(name) "opencv_" name CV_VERSION_ID
#endif

#pragma comment(lib, cvLIB("world"))
#pragma comment(lib, "UpixNetwork.lib")

#define MODEL_PATH "../../model/"
#define DATA_PATH "../../data/"

#define VIDEO_PATH "D:\\data\\423001.avi"
#define OUTPUT_VIDEO_PATH "D:\\data\\shabao_video_detect.mp4"

static const int INPUT_W = 192;
static const int INPUT_H = 192;
static const int CLASS_COUNT = 3;
static const char* CLASS_NAMES[] = { "red","blue","green" };

struct DetBox {
    int cls;
    float score;
    float x1, y1, x2, y2;
};

static cv::Scalar color_for_class(int cls) {
    switch (cls) {
    case 0: return cv::Scalar(0, 0, 255);
    case 1: return cv::Scalar(255, 0, 0);
    case 2: return cv::Scalar(0, 255, 0);
    default: return cv::Scalar(0, 255, 255);
    }
}

static float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }

static void softmax10_f(const float* src, float* dst) {
    float maxv = src[0];
    for (int i = 1; i < 10; i++) if (src[i] > maxv) maxv = src[i];
    float sum = 0.0f;
    for (int i = 0; i < 10; i++) { dst[i] = expf(src[i] - maxv); sum += dst[i]; }
    for (int i = 0; i < 10; i++) dst[i] /= sum;
}

static float dfl_expect10_f(const float* src) {
    float sm[10]; softmax10_f(src, sm);
    float v = 0.0f;
    for (int i = 0; i < 10; i++) v += sm[i] * (float)i;
    return v;
}

// WHC layout
static float get_cwh_f(const float* data, int w, int h, int c, int x, int y, int ch)
{
    return data[ch * w * h + y * w + x];
}

static void decode_one_head_float(const float* score_data, const float* bbox_data, int w, int h, int stride, float conf_thres, std::vector<DetBox>& dets) {
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int best_cls = -1; float best_score = 0.0f;
            for (int cls = 0; cls < CLASS_COUNT; cls++) {
                float raw = get_cwh_f(score_data, w, h, CLASS_COUNT, x, y, cls);
                float score = sigmoid_f(raw);
                if (score > best_score) { best_score = score; best_cls = cls; }
            }
            if (best_cls < 0 || best_score < conf_thres) continue;

            float tmp[10];
            for (int i = 0; i < 10; i++) tmp[i] = get_cwh_f(bbox_data, w, h, 40, x, y, i); float l = dfl_expect10_f(tmp);
            for (int i = 0; i < 10; i++) tmp[i] = get_cwh_f(bbox_data, w, h, 40, x, y, 10 + i); float t = dfl_expect10_f(tmp);
            for (int i = 0; i < 10; i++) tmp[i] = get_cwh_f(bbox_data, w, h, 40, x, y, 20 + i); float r = dfl_expect10_f(tmp);
            for (int i = 0; i < 10; i++) tmp[i] = get_cwh_f(bbox_data, w, h, 40, x, y, 30 + i); float b = dfl_expect10_f(tmp);

            float ax = (float)x + 0.5f, ay = (float)y + 0.5f;
            float x1 = (ax - l) * stride, y1 = (ay - t) * stride, x2 = (ax + r) * stride, y2 = (ay + b) * stride;

            if (x2 <= x1 || y2 <= y1) continue;
            DetBox d; d.cls = best_cls; d.score = best_score; d.x1 = x1; d.y1 = y1; d.x2 = x2; d.y2 = y2;
            dets.push_back(d);
        }
    }
}

static float iou_det(const DetBox& a, const DetBox& b) {
    float ix1 = std::max(a.x1, b.x1), iy1 = std::max(a.y1, b.y1);
    float ix2 = std::min(a.x2, b.x2), iy2 = std::min(a.y2, b.y2);
    float iw = std::max(0.0f, ix2 - ix1), ih = std::max(0.0f, iy2 - iy1);
    float inter = iw * ih;
    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1), area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    float uni = area_a + area_b - inter; if (uni <= 0.0f) return 0.0f;
    return inter / uni;
}

static std::vector<DetBox> nms_det(std::vector<DetBox>& dets, float iou_thres) {
    std::sort(dets.begin(), dets.end(), [](const DetBox& a, const DetBox& b) {return a.score > b.score; });
    std::vector<DetBox> keep;
    for (const auto& d : dets) {
        bool drop = false;
        for (const auto& k : keep) {
            if (d.cls == k.cls && iou_det(d, k) > iou_thres) { drop = true; break; }
        }
        if (!drop) keep.push_back(d);
    }
    return keep;
}

static float get_whc_debug(const float* data, int w, int h, int c, int x, int y, int ch)
{
    return data[(y * w + x) * c + ch];
}

static float get_cwh_debug(const float* data, int w, int h, int c, int x, int y, int ch)
{
    return data[ch * w * h + y * w + x];
}

static float sigmoid_debug(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

static void debug_score_cells(UpixNetwork& net)
{
    auto score48 = net.output(59);  // 48x48x3

    printf("\n===== debug score layer 59, expected high cells from ONNX =====\n");
    printf("layer59 shape: w=%d h=%d c=%d whc=%d\n",
        score48.w, score48.h, score48.c, score48.whc);

    struct Cell {
        int x;
        int y;
        int cls;
        const char* note;
    };

    // 这些点来自 ONNX top dets：
    // idx=1329 -> head48 idx=1329-576=753 -> y=15 x=33, red score≈0.8159
    // idx=1275 -> head48 idx=699 -> y=14 x=27, red score≈0.7256
    // idx=1036 -> head48 idx=460 -> y=9 x=28, blue score≈0.7242
    // idx=2707 -> head48 idx=2131 -> y=44 x=19, blue score≈0.6440
    Cell cells[] = {
        {33, 15, 0, "red top1, ONNX score about 0.8159"},
        {27, 14, 0, "red top2, ONNX score about 0.7256"},
        {28,  9, 1, "blue top1, ONNX score about 0.7242"},
        {19, 44, 1, "blue bottom, ONNX score about 0.6440"},
    };

    for (auto& cell : cells) {
        float raw_whc = get_whc_debug(score48.fptr, score48.w, score48.h, score48.c, cell.x, cell.y, cell.cls);
        float raw_cwh = get_cwh_debug(score48.fptr, score48.w, score48.h, score48.c, cell.x, cell.y, cell.cls);

        printf(
            "cell x=%d y=%d cls=%d | %s\n"
            "  WHC raw=%g sigmoid=%g\n"
            "  CWH raw=%g sigmoid=%g\n",
            cell.x,
            cell.y,
            cell.cls,
            cell.note,
            raw_whc,
            sigmoid_debug(raw_whc),
            raw_cwh,
            sigmoid_debug(raw_cwh)
        );
    }
}

static std::vector<DetBox> decode_shabao_float(UpixNetwork& net, float conf_thres) {
    std::vector<DetBox> dets;
    auto score24 = net.output(43); auto bbox24 = net.output(48);
    decode_one_head_float(score24.fptr, bbox24.fptr, score24.w, score24.h, 8, conf_thres, dets);
    auto score48 = net.output(59); auto bbox48 = net.output(64);
    decode_one_head_float(score48.fptr, bbox48.fptr, score48.w, score48.h, 4, conf_thres, dets);
    return nms_det(dets, 0.5f);
}

static void draw_detboxes(cv::Mat& img_bgr, const std::vector<DetBox>& dets, const char* title) {
    printf("\n----- %s result: %d boxes -----\n", title, (int)dets.size());
    for (const auto& d : dets) {
        int x1 = std::max(0, std::min((int)d.x1, img_bgr.cols - 1));
        int y1 = std::max(0, std::min((int)d.y1, img_bgr.rows - 1));
        int x2 = std::max(0, std::min((int)d.x2, img_bgr.cols - 1));
        int y2 = std::max(0, std::min((int)d.y2, img_bgr.rows - 1));
        if (x2 <= x1 || y2 <= y1) continue;
        cv::Scalar color = color_for_class(d.cls);
        cv::rectangle(img_bgr, cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)), color, 2);
        char label[128]; sprintf_s(label, "%s %.2f", CLASS_NAMES[d.cls], d.score);
        cv::putText(img_bgr, label, cv::Point(x1, std::max(15, y1 - 5)), cv::FONT_HERSHEY_SIMPLEX, 0.45, color, 1);
        printf("cls=%d(%s), score=%g, xyxy=(%g,%g,%g,%g)\n", d.cls, CLASS_NAMES[d.cls], d.score, d.x1, d.y1, d.x2, d.y2);
    }
}

// ============================================================
// 跟踪 + 四边形外到内计数
// ============================================================

struct CountPoint {
    int x;
    int y;
};

struct CountEvent {
    int index;
    int bag_uid;
    int cls_id;
    int frame_idx;
    float sec;
};

struct BagTrackCpp {
    int bag_uid = 0;
    int cls_id = -1;

    float x1 = 0;
    float y1 = 0;
    float x2 = 0;
    float y2 = 0;

    float cx = 0;
    float cy = 0;
    float vx = 0;
    float vy = 0;

    int first_frame = 0;
    int last_frame = 0;
    int seen_frames = 0;
    int missed_frames = 0;

    bool crossed_quad = false;
    bool inside_quad = false;
    bool armed_for_count = false;
    bool valid = false;
};

struct BagCounterCpp {
    int total_enter = 0;
    int count_by_cls[3] = { 0, 0, 0 };

    float match_distance = 90.0f;
    float reid_distance = 120.0f;
    int max_missed = 12;
    int next_uid = 1;

    std::vector<CountPoint> quad;
    std::vector<BagTrackCpp> tracks;
};

static float dist2d_f(float x1, float y1, float x2, float y2)
{
    float dx = x1 - x2;
    float dy = y1 - y2;
    return sqrtf(dx * dx + dy * dy);
}

static void det_center(const DetBox& d, float& cx, float& cy)
{
    cx = (d.x1 + d.x2) * 0.5f;
    cy = (d.y1 + d.y2) * 0.5f;
}

static bool point_on_segment_f(
    float px, float py,
    float x1, float y1,
    float x2, float y2
)
{
    float cross = fabsf((py - y1) * (x2 - x1) - (px - x1) * (y2 - y1));
    if (cross > 1e-4f) {
        return false;
    }

    float dot = (px - x1) * (px - x2) + (py - y1) * (py - y2);
    return dot <= 1e-4f;
}

static bool point_in_polygon_f(float px, float py, const std::vector<CountPoint>& poly)
{
    int n = (int)poly.size();
    if (n < 3) {
        return false;
    }

    for (int i = 0; i < n; ++i) {
        const CountPoint& a = poly[i];
        const CountPoint& b = poly[(i + 1) % n];

        if (point_on_segment_f(px, py, (float)a.x, (float)a.y, (float)b.x, (float)b.y)) {
            return true;
        }
    }

    bool inside = false;

    for (int i = 0; i < n; ++i) {
        float x1 = (float)poly[i].x;
        float y1 = (float)poly[i].y;
        float x2 = (float)poly[(i + 1) % n].x;
        float y2 = (float)poly[(i + 1) % n].y;

        if ((py > std::min(y1, y2)) &&
            (py <= std::max(y1, y2)) &&
            (px <= std::max(x1, x2))) {

            float xinters = x1;

            if (fabsf(y1 - y2) > 1e-6f) {
                xinters = (py - y1) * (x2 - x1) / (y2 - y1) + x1;
            }

            if (fabsf(x1 - x2) < 1e-6f || px <= xinters) {
                inside = !inside;
            }
        }
    }

    return inside;
}

static bool check_enter_quad(BagTrackCpp& tr, const std::vector<CountPoint>& quad)
{
    bool is_inside = point_in_polygon_f(tr.cx, tr.cy, quad);

    // 新轨迹第一帧：只初始化，不计数。
    if (tr.seen_frames <= 1) {
        tr.inside_quad = is_inside;
        tr.armed_for_count = !is_inside;
        return false;
    }

    // 如果第一次出现就在区域内，永远不计数，防止区域内丢失后重复加。
    if (!tr.armed_for_count) {
        tr.inside_quad = is_inside;
        return false;
    }

    // 只有从外到内才计数。
    bool entered = (!tr.inside_quad) && is_inside;
    tr.inside_quad = is_inside;

    return entered;
}

static void init_counter(
    BagCounterCpp& counter,
    const std::vector<CountPoint>& quad,
    float match_distance,
    float reid_distance,
    int max_missed
)
{
    counter.total_enter = 0;
    counter.count_by_cls[0] = 0;
    counter.count_by_cls[1] = 0;
    counter.count_by_cls[2] = 0;

    counter.match_distance = match_distance;
    counter.reid_distance = reid_distance;
    counter.max_missed = max_missed;
    counter.next_uid = 1;
    counter.quad = quad;
    counter.tracks.clear();
}

static void create_track(BagCounterCpp& counter, const DetBox& d, int frame_idx)
{
    float cx = 0.0f;
    float cy = 0.0f;
    det_center(d, cx, cy);

    BagTrackCpp tr;
    tr.bag_uid = counter.next_uid++;
    tr.cls_id = d.cls;

    tr.x1 = d.x1;
    tr.y1 = d.y1;
    tr.x2 = d.x2;
    tr.y2 = d.y2;

    tr.cx = cx;
    tr.cy = cy;
    tr.vx = 0.0f;
    tr.vy = 0.0f;

    tr.first_frame = frame_idx;
    tr.last_frame = frame_idx;
    tr.seen_frames = 1;
    tr.missed_frames = 0;

    tr.crossed_quad = false;
    tr.inside_quad = false;
    tr.armed_for_count = false;
    tr.valid = true;

    counter.tracks.push_back(tr);
}

static void update_track(BagTrackCpp& tr, const DetBox& d, int frame_idx)
{
    float new_cx = 0.0f;
    float new_cy = 0.0f;
    det_center(d, new_cx, new_cy);

    tr.vx = new_cx - tr.cx;
    tr.vy = new_cy - tr.cy;

    tr.cx = new_cx;
    tr.cy = new_cy;

    tr.x1 = d.x1;
    tr.y1 = d.y1;
    tr.x2 = d.x2;
    tr.y2 = d.y2;

    tr.last_frame = frame_idx;
    tr.seen_frames++;
    tr.missed_frames = 0;
}

static std::vector<CountEvent> update_counter_tracks(
    BagCounterCpp& counter,
    const std::vector<DetBox>& dets,
    int frame_idx,
    double fps
)
{
    std::vector<CountEvent> events;

    // 1. 简单常速度预测。
    for (auto& tr : counter.tracks) {
        if (!tr.valid) {
            continue;
        }

        tr.cx += tr.vx;
        tr.cy += tr.vy;
        tr.missed_frames++;
    }

    std::vector<int> det_used(dets.size(), 0);

    // 2. 贪心最近距离匹配，同类别才匹配。
    for (int i = 0; i < (int)dets.size(); ++i) {
        const DetBox& d = dets[i];

        float dcx = 0.0f;
        float dcy = 0.0f;
        det_center(d, dcx, dcy);

        int best_j = -1;
        float best_dist = 1e9f;

        for (int j = 0; j < (int)counter.tracks.size(); ++j) {
            BagTrackCpp& tr = counter.tracks[j];

            if (!tr.valid) {
                continue;
            }

            if (tr.cls_id != d.cls) {
                continue;
            }

            float threshold = (tr.missed_frames <= 2)
                ? counter.match_distance
                : counter.reid_distance;

            float dd = dist2d_f(dcx, dcy, tr.cx, tr.cy);

            if (dd <= threshold && dd < best_dist) {
                best_dist = dd;
                best_j = j;
            }
        }

        if (best_j >= 0) {
            update_track(counter.tracks[best_j], d, frame_idx);
            det_used[i] = 1;
        }
    }

    // 3. 未匹配检测框新建轨迹。
    for (int i = 0; i < (int)dets.size(); ++i) {
        if (!det_used[i]) {
            create_track(counter, dets[i], frame_idx);
        }
    }

    // 4. 删除丢失太久的轨迹。
    for (auto& tr : counter.tracks) {
        if (!tr.valid) {
            continue;
        }

        if (tr.missed_frames > counter.max_missed) {
            tr.valid = false;
        }
    }

    // 5. 外到内计数。
    for (auto& tr : counter.tracks) {
        if (!tr.valid) {
            continue;
        }

        if (tr.crossed_quad) {
            continue;
        }

        if (check_enter_quad(tr, counter.quad)) {
            tr.crossed_quad = true;

            counter.total_enter++;
            if (tr.cls_id >= 0 && tr.cls_id < 3) {
                counter.count_by_cls[tr.cls_id]++;
            }

            CountEvent ev;
            ev.index = counter.total_enter;
            ev.bag_uid = tr.bag_uid;
            ev.cls_id = tr.cls_id;
            ev.frame_idx = frame_idx;
            ev.sec = fps > 0.0 ? (float)(frame_idx / fps) : -1.0f;
            events.push_back(ev);

            printf(
                "[Frame %d] Enter quad: ID=%d, cls=%d(%s), total=%d, red=%d, blue=%d, green=%d\n",
                frame_idx,
                tr.bag_uid,
                tr.cls_id,
                CLASS_NAMES[tr.cls_id],
                counter.total_enter,
                counter.count_by_cls[0],
                counter.count_by_cls[1],
                counter.count_by_cls[2]
            );
        }
    }

    return events;
}

static void draw_quad_count_area(cv::Mat& img_bgr, const std::vector<CountPoint>& quad)
{
    if (quad.size() < 3) {
        return;
    }

    std::vector<cv::Point> pts;
    for (const auto& p : quad) {
        pts.push_back(cv::Point(p.x, p.y));
    }

    cv::polylines(img_bgr, pts, true, cv::Scalar(0, 220, 220), 2);
}

static void draw_counter_tracks(cv::Mat& img_bgr, const BagCounterCpp& counter)
{
    for (const auto& tr : counter.tracks) {
        if (!tr.valid) {
            continue;
        }

        cv::Scalar color = color_for_class(tr.cls_id);

        int x1 = std::max(0, std::min((int)tr.x1, img_bgr.cols - 1));
        int y1 = std::max(0, std::min((int)tr.y1, img_bgr.rows - 1));
        int x2 = std::max(0, std::min((int)tr.x2, img_bgr.cols - 1));
        int y2 = std::max(0, std::min((int)tr.y2, img_bgr.rows - 1));

        if (x2 > x1 && y2 > y1) {
            cv::rectangle(img_bgr, cv::Rect(cv::Point(x1, y1), cv::Point(x2, y2)), color, 2);
        }

        int cx = std::max(0, std::min((int)tr.cx, img_bgr.cols - 1));
        int cy = std::max(0, std::min((int)tr.cy, img_bgr.rows - 1));

        cv::circle(img_bgr, cv::Point(cx, cy), 4, cv::Scalar(255, 255, 255), -1);

        char text[128];
        sprintf_s(text, "ID:%d %s", tr.bag_uid, CLASS_NAMES[tr.cls_id]);

        cv::putText(
            img_bgr,
            text,
            cv::Point(x1, std::max(16, y1 - 6)),
            cv::FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1
        );
    }
}

static void draw_counter_panel(cv::Mat& img_bgr, const BagCounterCpp& counter)
{
    cv::rectangle(img_bgr, cv::Rect(0, 0, 120, 72), cv::Scalar(30, 30, 30), -1);

    char text[128];

    sprintf_s(text, "Total:%d", counter.total_enter);
    cv::putText(img_bgr, text, cv::Point(6, 16),
        cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 255, 255), 1);

    sprintf_s(text, "Red:%d", counter.count_by_cls[0]);
    cv::putText(img_bgr, text, cv::Point(6, 34),
        cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 255), 1);

    sprintf_s(text, "Blue:%d", counter.count_by_cls[1]);
    cv::putText(img_bgr, text, cv::Point(6, 52),
        cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 100, 0), 1);

    sprintf_s(text, "Green:%d", counter.count_by_cls[2]);
    cv::putText(img_bgr, text, cv::Point(6, 70),
        cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 200, 0), 1);
}

// ============================================================
// 运行前手动画四边形区域
// 左键点 4 个点，按 c 确认，按 r 重选，按 q 取消
// ============================================================

static std::vector<CountPoint> g_select_points;
static cv::Mat g_select_base;
static cv::Mat g_select_show;

static void redraw_select_window(const char* win_name)
{
    g_select_show = g_select_base.clone();

    for (int i = 0; i < (int)g_select_points.size(); ++i) {
        cv::Point p(g_select_points[i].x, g_select_points[i].y);
        cv::circle(g_select_show, p, 4, cv::Scalar(0, 255, 0), -1);

        char text[32];
        sprintf_s(text, "%d", i + 1);
        cv::putText(
            g_select_show,
            text,
            cv::Point(p.x + 5, p.y - 5),
            cv::FONT_HERSHEY_SIMPLEX,
            0.45,
            cv::Scalar(0, 255, 0),
            1
        );
    }

    if (g_select_points.size() >= 2) {
        std::vector<cv::Point> pts;
        for (const auto& p : g_select_points) {
            pts.push_back(cv::Point(p.x, p.y));
        }

        bool closed = g_select_points.size() >= 3;
        cv::polylines(g_select_show, pts, closed, cv::Scalar(0, 255, 0), 1);
    }

    cv::putText(
        g_select_show,
        "Left click polygon points, c=confirm, r=reset, backspace=undo, q=quit",
        cv::Point(6, 18),
        cv::FONT_HERSHEY_SIMPLEX,
        0.38,
        cv::Scalar(0, 255, 255),
        1
    );

    cv::imshow(win_name, g_select_show);
}

static void on_select_polygon_mouse(int event, int x, int y, int flags, void* userdata)
{
    const char* win_name = (const char*)userdata;

    if (event == cv::EVENT_LBUTTONDOWN) {
        CountPoint p;
        p.x = x;
        p.y = y;
        g_select_points.push_back(p);
        redraw_select_window(win_name);
    }
}

static bool select_polygon_before_run(cv::VideoCapture& cap, std::vector<CountPoint>& poly_points)
{
    cv::Mat first_frame;
    if (!cap.read(first_frame) || first_frame.empty()) {
        printf("read first frame failed for polygon select\n");
        return false;
    }

    cv::resize(first_frame, g_select_base, cv::Size(INPUT_W, INPUT_H));

    const char* win_name = "select count polygon";
    g_select_points.clear();

    cv::namedWindow(win_name);
    cv::setMouseCallback(win_name, on_select_polygon_mouse, (void*)win_name);

    redraw_select_window(win_name);

    while (true) {
        int key = cv::waitKey(20);

        if (key == 'c' || key == 'C') {
            if ((int)g_select_points.size() >= 3) {
                poly_points = g_select_points;
                cv::destroyWindow(win_name);

                // 选区读掉了第一帧，这里回到第 0 帧。
                cap.set(cv::CAP_PROP_POS_FRAMES, 0);

                printf("polygon selected, points=%d:\n", (int)poly_points.size());
                for (int i = 0; i < (int)poly_points.size(); ++i) {
                    printf("  p%d = (%d, %d)\n", i + 1, poly_points[i].x, poly_points[i].y);
                }

                return true;
            }
            else {
                printf("need at least 3 points, current=%d\n", (int)g_select_points.size());
            }
        }
        else if (key == 'r' || key == 'R') {
            g_select_points.clear();
            redraw_select_window(win_name);
        }
        else if (key == 8 || key == 127) {
            // Backspace 删除最后一个点
            if (!g_select_points.empty()) {
                g_select_points.pop_back();
                redraw_select_window(win_name);
            }
        }
        else if (key == 'q' || key == 'Q' || key == 27) {
            cv::destroyWindow(win_name);
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            return false;
        }
    }
}

static int get_cwh_i(const int* data, int w, int h, int c, int x, int y, int ch)
{
    return data[ch * w * h + y * w + x];
}

static void softmax10_i_to_float(const int* src, float* dst)
{
    float tmp[10];

    float maxv = (float)src[0];
    for (int i = 1; i < 10; i++) {
        if ((float)src[i] > maxv) {
            maxv = (float)src[i];
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < 10; i++) {
        tmp[i] = expf((float)src[i] - maxv);
        sum += tmp[i];
    }

    if (sum <= 0.0f) {
        sum = 1.0f;
    }

    for (int i = 0; i < 10; i++) {
        dst[i] = tmp[i] / sum;
    }
}

static float dfl_expect10_i(const int* src)
{
    float sm[10];
    softmax10_i_to_float(src, sm);

    float v = 0.0f;
    for (int i = 0; i < 10; i++) {
        v += sm[i] * (float)i;
    }

    return v;
}

static void decode_one_head_int(
    const int* score_data,
    const int* bbox_data,
    int w,
    int h,
    int stride,
    float conf_thres,
    std::vector<DetBox>& dets
)
{
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {

            int best_cls = -1;
            float best_score = 0.0f;

            for (int cls = 0; cls < CLASS_COUNT; cls++) {
                int raw_i = get_cwh_i(score_data, w, h, CLASS_COUNT, x, y, cls);

                // int 输出一般是 Q10 缩放，先按 /1024 解
                float score = 1.0f / (1.0f + expf(-(float)raw_i / 1024.0f));

                if (score > best_score) {
                    best_score = score;
                    best_cls = cls;
                }
            }

            if (best_cls < 0 || best_score < conf_thres) {
                continue;
            }

            int tmp_i[10];

            for (int i = 0; i < 10; i++) {
                tmp_i[i] = get_cwh_i(bbox_data, w, h, 40, x, y, 0 + i);
            }
            float l = dfl_expect10_i(tmp_i);

            for (int i = 0; i < 10; i++) {
                tmp_i[i] = get_cwh_i(bbox_data, w, h, 40, x, y, 10 + i);
            }
            float t = dfl_expect10_i(tmp_i);

            for (int i = 0; i < 10; i++) {
                tmp_i[i] = get_cwh_i(bbox_data, w, h, 40, x, y, 20 + i);
            }
            float r = dfl_expect10_i(tmp_i);

            for (int i = 0; i < 10; i++) {
                tmp_i[i] = get_cwh_i(bbox_data, w, h, 40, x, y, 30 + i);
            }
            float b = dfl_expect10_i(tmp_i);

            float anchor_x = (float)x + 0.5f;
            float anchor_y = (float)y + 0.5f;

            float x1 = (anchor_x - l) * stride;
            float y1 = (anchor_y - t) * stride;
            float x2 = (anchor_x + r) * stride;
            float y2 = (anchor_y + b) * stride;

            if (x2 <= x1 || y2 <= y1) {
                continue;
            }

            DetBox d;
            d.cls = best_cls;
            d.score = best_score;
            d.x1 = x1;
            d.y1 = y1;
            d.x2 = x2;
            d.y2 = y2;

            dets.push_back(d);
        }
    }
}

static std::vector<DetBox> decode_shabao_int(UpixNetwork& net, float conf_thres)
{
    std::vector<DetBox> dets;

    auto score24 = net.output(43);
    auto bbox24 = net.output(48);

    decode_one_head_int(
        score24.iptr,
        bbox24.iptr,
        score24.w,
        score24.h,
        8,
        conf_thres,
        dets
    );

    auto score48 = net.output(59);
    auto bbox48 = net.output(64);

    decode_one_head_int(
        score48.iptr,
        bbox48.iptr,
        score48.w,
        score48.h,
        4,
        conf_thres,
        dets
    );

    return nms_det(dets, 0.5f);
}


static void run_image_int_rgb(UpixNetwork* net, cv::Mat& img_rgb)
{
    static std::vector<int> input_chw;
    input_chw.resize(INPUT_W * INPUT_H * 3);

    for (int y = 0; y < INPUT_H; y++) {
        const cv::Vec3b* row = img_rgb.ptr<cv::Vec3b>(y);

        for (int x = 0; x < INPUT_W; x++) {
            int idx = y * INPUT_W + x;

            // img_rgb 是 RGB 顺序，int 路径不除 255
            input_chw[0 * INPUT_W * INPUT_H + idx] = row[x][0];  // R
            input_chw[1 * INPUT_W * INPUT_H + idx] = row[x][1];  // G
            input_chw[2 * INPUT_W * INPUT_H + idx] = row[x][2];  // B
        }
    }

    net->forward(input_chw.data());
}

static void run_image_float_rgb(UpixNetwork* net, cv::Mat& img_rgb)
{
    static std::vector<float> input_chw;
    input_chw.resize(INPUT_W * INPUT_H * 3);

    for (int y = 0; y < INPUT_H; y++) {
        const cv::Vec3b* row = img_rgb.ptr<cv::Vec3b>(y);

        for (int x = 0; x < INPUT_W; x++) {
            int idx = y * INPUT_W + x;

            // img_rgb 是 RGB 顺序
            input_chw[0 * INPUT_W * INPUT_H + idx] = row[x][0] / 255.0f;  // R
            input_chw[1 * INPUT_W * INPUT_H + idx] = row[x][1] / 255.0f;  // G
            input_chw[2 * INPUT_W * INPUT_H + idx] = row[x][2] / 255.0f;  // B
        }
    }

    net->forward(input_chw.data());
}

static bool load_int_net(UpixNetwork& net)
{
    if (!net.loadNetwork(MODEL_PATH "shabao_192x192.network.csv")) {
        printf("load int network failed\n");
        return false;
    }

    if (!net.loadWeight(MODEL_PATH "shabao_192x192.weights.txt")) {
        printf("load int weight failed\n");
        return false;
    }

    if (!net.loadQuant(MODEL_PATH "shabao_192x192.maxmin.txt")) {
        printf("load quant failed\n");
        return false;
    }

    return true;
}

static void test_shabao_video_192x192(const char* video_path, const char* output_path)
{
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        printf("open video failed: %s\n", video_path);
        return;
    }

    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0.0 || fps > 1000.0) {
        fps = 25.0;
    }

    printf("video opened: %s\n", video_path);
    printf("fps = %.2f\n", fps);
    // 计数区域，当前是 192x192 坐标。
    // 你需要根据实际视频画面调整这 4 个点。
    std::vector<CountPoint> poly_points;

    if (!select_polygon_before_run(cap, poly_points)) {
        printf("polygon select canceled\n");
        return;
    }

    BagCounterCpp counter;
    init_counter(
        counter,
        poly_points,
        90.0f,
        120.0f,
        12
    );
    UpixNetwork net;

#if 1
    // 当前先跑 int 量化网络，更接近最终 C/嵌入式部署路径。
    if (!load_int_net(net)) {
        printf("load int net failed\n");
        return;
    }
    printf("int model loaded ok\n");
#else
    // 如果你想先对齐 float，把上面的 #if 1 改成 #if 0。
    if (!net.loadNetwork(MODEL_PATH "shabao_192x192.network.csv")) {
        printf("load float network failed\n");
        return;
    }

    if (!net.loadWeight(MODEL_PATH "shabao_192x192.weights.txt")) {
        printf("load float weight failed\n");
        return;
    }
    printf("float model loaded ok\n");
#endif

    cv::VideoWriter writer;
    bool save_video = (output_path != nullptr && output_path[0] != '\0');

    if (save_video) {
        int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        writer.open(output_path, fourcc, fps, cv::Size(INPUT_W, INPUT_H), true);
        if (!writer.isOpened()) {
            printf("open output video failed: %s\n", output_path);
            save_video = false;
        }
        else {
            printf("output video: %s\n", output_path);
        }
    }

    int frame_idx = 0;
    float conf_thres = 0.15f;

    while (true) {
        cv::Mat frame_bgr;
        if (!cap.read(frame_bgr) || frame_bgr.empty()) {
            printf("video end\n");
            break;
        }

        cv::Mat img_bgr;
        cv::resize(frame_bgr, img_bgr, cv::Size(INPUT_W, INPUT_H));

        cv::Mat img_rgb;
        cv::cvtColor(img_bgr, img_rgb, cv::COLOR_BGR2RGB);

#if 1
        run_image_int_rgb(&net, img_rgb);
        std::vector<DetBox> dets = decode_shabao_int(net, conf_thres);
        const char* title = "int";
#else
        run_image_float_rgb(&net, img_rgb);
        std::vector<DetBox> dets = decode_shabao_float(net, conf_thres);
        const char* title = "float";
#endif

        std::vector<CountEvent> events = update_counter_tracks(
            counter,
            dets,
            frame_idx,
            fps
        );

        cv::Mat show = img_bgr.clone();

        draw_quad_count_area(show, counter.quad);
        draw_counter_tracks(show, counter);
        draw_counter_panel(show, counter);

        char info[128];
        sprintf_s(info, "frame:%d det:%d event:%d", frame_idx, (int)dets.size(), (int)events.size());
        cv::putText(
            show,
            info,
            cv::Point(8, 90),
            cv::FONT_HERSHEY_SIMPLEX,
            0.45,
            cv::Scalar(0, 255, 255),
            1
        );

        cv::imshow("shabao video detect 192x192", show);

        if (save_video) {
            writer.write(show);
        }

        int key = cv::waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q') {
            printf("user quit\n");
            break;
        }

        frame_idx++;
    }

    cap.release();

    if (writer.isOpened()) {
        writer.release();
    }
    printf("\n================ final count ================\n");
    printf("Total: %d\n", counter.total_enter);
    printf("Red  : %d\n", counter.count_by_cls[0]);
    printf("Blue : %d\n", counter.count_by_cls[1]);
    printf("Green: %d\n", counter.count_by_cls[2]);
    printf("=============================================\n");
    cv::destroyAllWindows();
}

int main(int argc, char** argv)
{
    const char* video_path = VIDEO_PATH;
    const char* output_path = OUTPUT_VIDEO_PATH;

    if (argc >= 2) {
        video_path = argv[1];
    }

    if (argc >= 3) {
        output_path = argv[2];
    }

    test_shabao_video_192x192(video_path, output_path);

    system("pause");
    return 0;
}