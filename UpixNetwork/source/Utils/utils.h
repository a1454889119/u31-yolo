#ifndef __UPIX_NETWORK_UTILS_H__
#define __UPIX_NETWORK_UTILS_H__

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
class UpixNetwork;

std::vector<std::string> load_output_names(const char* filename);

std::vector<std::string> get_img_full_names(const char* imgpath);

void run_image_float(UpixNetwork* net, cv::Mat& img);
void run_image_int(UpixNetwork* net, cv::Mat& img);

bool dump_maxmin(
	const char* networkfilename, 
	const char* weightsfilename,
	const char* imagepath, 
	const char* outputfilename);

typedef struct stDecodeInfo {
	int score_layer;
	int rect_layer;
	int scale;
}DecodeInfo;

void setDecodeInfo(int type_count, float thres, const std::vector<DecodeInfo>& dec_infos);
void draw_rects_float(UpixNetwork* net, cv::Mat& img, int color_idx);
void draw_rects_int(UpixNetwork* net, cv::Mat& img, int color_idx);

void dump_int16_result(UpixNetwork* netq, const char* outpath);
void dump_int8_result(UpixNetwork& netq, const char* outpath);

#endif // !__UPIX_NETWORK_UTILS_H__
