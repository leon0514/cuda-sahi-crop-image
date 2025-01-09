#ifndef SLICE_HPP__
#define SLICE_HPP__

#include "opencv2/opencv.hpp"
#include "common/image.hpp"
#include "common/memory.hpp"
#include <vector>

namespace slice
{

struct SlicedImageData{
    cv::Mat image;
    float x;
    float y;
    float w;
    float h;
};

class SliceImage{
private:
    tensor::Memory<unsigned char> input_image_;
    tensor::Memory<unsigned char> output_images_;
public:
    std::vector<SlicedImageData> slice(
        const tensor::Image& image, 
        const int slice_num_h, 
        const int slice_num_v,
        const float overlap_ratio,
        void* stream=nullptr);

};



}


#endif