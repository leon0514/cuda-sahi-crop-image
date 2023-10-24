#ifndef SLICE_H__
#define SLICE_H__
#include <cuda_runtime.h>
#include <stdint.h>
#include <vector>
#include "opencv2/opencv.hpp"
/**
 * @brief 图片切图
 * @param data 图片的gpu指针
 * @param slice_images 切出来的图以vector保存Mat
 * @param crop_size 切出来的图相对于原始图片的坐标起始点
 * @param width 图片宽度
 * @param height 图片高度
 * @param slice_num_h 水平部分切割数量
 * @param slice_nms_v 垂直部分切割数量
 * @param overlap_ratio 重叠区域比例，计算重叠区域时按照宽高最大值来计算
*/
void slice(const uint8_t* data,
            std::vector<cv::Mat>& slice_images,
            std::vector<cv::Rect_<float>>& crop_size,
            const int width,
            const int height,
            const int slice_num_h, 
            const int slice_num_v,
            const float overlap_ratio);

#define MAX(a, b) (a) > (b) ? (a) : (b)

#endif