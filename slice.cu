#include "slice.h"

__global__ void slice_kernel(
  const uint8_t*  image,
  uint8_t*  outs,
  const int width,
  const int height,
  const int slice_width,
  const int slice_height,
  const int slice_num_h,
  const int slice_num_v,
  const int overlap_pixel)
{
    const int out_size = 3 * slice_width * slice_height;
    int dx = blockDim.x * blockIdx.x + threadIdx.x;
    int dy = blockDim.y * blockIdx.y + threadIdx.y;
    if (dx >= width || dy >= height)
    {
        return;
    }
    int offset = dy * width + dx;
    uint8_t b = image[3 * offset + 0];
    uint8_t g = image[3 * offset + 1];
    uint8_t r = image[3 * offset + 2];
    for (int i = 0; i < slice_num_h; i++)
    {
        int sdx_start = MAX(0, i * slice_width - overlap_pixel);
        int sdx_end   = sdx_start + slice_width;
        for (int j = 0; j < slice_num_v; j++)
        {
            int sdy_start = MAX(0, j * slice_height - overlap_pixel);
            int sdy_end   = sdy_start + slice_height;
            if (dx >= sdx_start && dx < sdx_end && dy >= sdy_start && dy < sdy_end)
            {
                int image_id = i * slice_num_h + j;
                int sdx = dx - sdx_start;
                int sdy = dy - sdy_start;
                int soffset = sdy * slice_width + sdx;
                outs[image_id * out_size + 3 * soffset + 0] = b;
                outs[image_id * out_size + 3 * soffset + 1] = g;
                outs[image_id * out_size + 3 * soffset + 2] = r;
            }
        }
    }
}

void slice(
  const uint8_t* data,
  std::vector<cv::Mat>& slice_images, 
  std::vector<cv::Rect_<float>>& crop_size,
  const int width,
  const int height,
  const int slice_num_h, 
  const int slice_num_v, 
  const float overlap_ratio)
{
    int slice_num = slice_num_h * slice_num_v;
    const int overlap_pixel = std::max(width, height) * overlap_ratio;
    const int slice_width = (width - overlap_pixel) / slice_num_h + overlap_pixel;
    const int slice_height = (height - overlap_pixel) / slice_num_v + overlap_pixel;
    int output_img_size = 3 * slice_width * slice_height;
    uint8_t* output_imgs_gpu;
    cudaMalloc((void**)&output_imgs_gpu, output_img_size * slice_num * sizeof(uint8_t));
    cudaMemset(output_imgs_gpu, 114, output_img_size * slice_num * sizeof(uint8_t));
    dim3 threads(32, 32);
    dim3 blocks((width + 31) / 32, (height + 31) / 32);

    slice_kernel<<<blocks, threads>>>(data, output_imgs_gpu, 
                                            width, height, 
                                            slice_width, slice_height, 
                                            slice_num_h, slice_num_v, overlap_pixel);
    cudaDeviceSynchronize();
    slice_images.resize(slice_num);
    crop_size.resize(slice_num);
    for (int i = 0; i < slice_num_h; i++)
    {
        int x = MAX(0, i * slice_width - overlap_pixel);
        for (int j = 0; j < slice_num_v; j++)
        {
            int y = MAX(0, j * slice_height - overlap_pixel);
            int image_id = i * slice_num_h + j;
            crop_size[image_id] = cv::Rect_<float>(cv::Point_<float>(x, y), cv::Point_<float>(x+slice_width, y+slice_height));
            slice_images[image_id] = cv::Mat(slice_height, slice_width, CV_8UC3);
            uint8_t* output_img_data = slice_images[image_id].ptr<uint8_t>();
            cudaMemcpy(output_img_data, output_imgs_gpu+image_id*output_img_size, output_img_size*sizeof(uint8_t), cudaMemcpyDeviceToHost);
        }
    }
    cudaFree(output_imgs_gpu);
}
