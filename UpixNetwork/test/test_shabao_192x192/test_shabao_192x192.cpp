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

static void test_shabao_192x192() {
    cv::Mat img_bgr = cv::imread(DATA_PATH "shabao_test.jpg");
    if (img_bgr.empty()) { printf("image not found\n"); return; }
    cv::resize(img_bgr, img_bgr, cv::Size(INPUT_W, INPUT_H));
    cv::Mat img_rgb; cv::cvtColor(img_bgr, img_rgb, cv::COLOR_BGR2RGB);

    UpixNetwork net_float;
    UpixNetwork net_int;

    if (!net_float.loadNetwork(MODEL_PATH "shabao_192x192.network.csv")) {
        printf("load float network failed\n");
        return;
    }

    if (!net_float.loadWeight(MODEL_PATH "shabao_192x192.weights.txt")) {
        printf("load float weight failed\n");
        return;
    }

    if (!load_int_net(net_int)) {
        return;
    }

    printf("model loaded ok\n");

    run_image_float_rgb(&net_float, img_rgb);
    run_image_int_rgb(&net_int, img_rgb);

    float conf_thres = 0.15f;

    std::vector<DetBox> dets_float = decode_shabao_float(net_float, conf_thres);
    std::vector<DetBox> dets_int = decode_shabao_int(net_int, conf_thres);

    cv::Mat show_float = img_bgr.clone();
    cv::Mat show_int = img_bgr.clone();

    draw_detboxes(show_float, dets_float, "float");
    draw_detboxes(show_int, dets_int, "int");

    cv::putText(
        show_float,
        "float",
        cv::Point(10, 20),
        cv::FONT_HERSHEY_SIMPLEX,
        0.6,
        cv::Scalar(0, 255, 255),
        2
    );

    cv::putText(
        show_int,
        "int",
        cv::Point(10, 20),
        cv::FONT_HERSHEY_SIMPLEX,
        0.6,
        cv::Scalar(0, 255, 255),
        2
    );

    cv::Mat show;
    cv::hconcat(show_float, show_int, show);

    cv::imshow("shabao detect: left=float, right=int", show);
    cv::imwrite("shabao_float_int_detect.png", show);
    cv::waitKey(0);
}

int main(int argc, char** argv) {
    test_shabao_192x192();
    system("pause");
    return 0;
}