#ifndef __UTIL_FUNC_H__
#define __UTIL_FUNC_H__

#include <vector>
#include <map>
#include <string>
#include <opencv2/opencv.hpp>
#include <S16Network/layers/S16CNN_Common.h>

class S16Network;

std::vector<std::string> get_img_full_names(const char* imgpath);

void get_input_data(const cv::Mat& img, short* input_data);

void image2bin(const cv::Mat& img, const char* filename);

std::vector<Quantize> load_scales(const char* filename);

void set_scales(S16Network* network, const std::vector<Quantize>& scales);

void quantlize_weights(
	S16Network* network,
	const char* in_weights_file,
	const char* in_maxmin_file, 
	const char* in_name_idx_file,
	const char* out_weights);

#endif // !__UTIL_FUNC_H__
