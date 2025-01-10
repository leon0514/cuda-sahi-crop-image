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

int calculateNumCuts(int dimension, int subDimension, float overlapRatio);

class SliceImage{
private:
    tensor::Memory<unsigned char> input_image_;
    tensor::Memory<unsigned char> output_images_;
public:
    std::vector<SlicedImageData> slice(
        const tensor::Image& image, 
        const int slice_width,
        const int slice_height, 
        const float overlap_width_ratio,
        const float overlap_height_ratio,
        void* stream=nullptr);
    
    std::vector<SlicedImageData> autoSlice(
        const tensor::Image& image, 
        void* stream=nullptr);
};



}


#endif