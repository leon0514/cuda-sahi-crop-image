#include "slice/slice.hpp"
#include "common/check.hpp"


static __global__ void slice_kernel(
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
    if (dx >= width || dy >= height || dx < 0 || dy < 0)
    {
        return;
    }
    int offset = dy * width + dx;
    uint8_t b = image[3 * offset + 0];
    uint8_t g = image[3 * offset + 1];
    uint8_t r = image[3 * offset + 2];


    // 定义共享内存，存储切片范围
    __shared__ int slice_range_h[32]; // 假设一个线程块最多处理 32 个切片（可以根据实际情况调整）
    __shared__ int slice_range_v[32]; 

    // 计算切片的起始和结束位置，并存储在共享内存中
    if (threadIdx.x < slice_num_h) 
    {
        // 这里计算start的时候必须分两行，先计算start，再取0和start的最大值
        int start = threadIdx.x * (slice_width - overlap_pixel);
        start = max(start, 0);
        int end = start + slice_width;
        slice_range_h[threadIdx.x * 2] = start;
        slice_range_h[threadIdx.x * 2 + 1] = end;
        
    }

    if (threadIdx.y < slice_num_v) {
        int start = threadIdx.y * (slice_height - overlap_pixel);
        start = max(start, 0);
        int end = start + slice_height;
        slice_range_v[threadIdx.y * 2] = start;
        slice_range_v[threadIdx.y * 2 + 1] = end;
        
    }
    __syncthreads();

    for (int i = 0; i < slice_num_h; i++)
    {
        int sdx_start = slice_range_h[i * 2];
        int sdx_end   = slice_range_h[i * 2 + 1];

        for (int j = 0; j < slice_num_v; j++)
        {
            int sdy_start = slice_range_v[j * 2];
            int sdy_end = slice_range_v[j * 2 + 1];   
            if (dx >= sdx_start && dx < sdx_end && dy >= sdy_start && dy < sdy_end)
            {
                int image_id = i * slice_num_v + j;
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

static void slice_plane(const uint8_t* image,
    uint8_t*  outs,
    const int width,
    const int height,
    const int slice_width,
    const int slice_height,
    const int slice_num_h,
    const int slice_num_v,
    const int overlap_pixel,
    void* stream=nullptr)
{
    cudaStream_t stream_ = (cudaStream_t)stream;
    dim3 block(32, 32);
    dim3 grid((width + 31) / 32, (height + 31) / 32);

    slice_kernel<<<grid, block, 0, stream_>>>(image, outs, 
                                    width, height, 
                                    slice_width, slice_height, 
                                    slice_num_h, slice_num_v, overlap_pixel);
}


namespace slice
{

std::vector<SlicedImageData> SliceImage::slice(
    const tensor::Image& image, 
    const int slice_num_h, 
    const int slice_num_v,
    const float overlap_ratio,
    void* stream)
{
    cudaStream_t stream_ = (cudaStream_t)stream;

    int width = image.width;
    int height = image.height;

    int slice_num = slice_num_h * slice_num_v;
    int overlap_pixel = std::min(width, height) * overlap_ratio;
    int slice_width = (width - overlap_pixel) / slice_num_h + overlap_pixel;
    int slice_height = (height - overlap_pixel) / slice_num_v + overlap_pixel;

    size_t size_image = 3 * width * height;
    size_t output_img_size = 3 * slice_width * slice_height;

    input_image_.gpu(size_image);
    // input_image_.cpu(size_image);

    output_images_.gpu(slice_num * output_img_size);
    // output_images_.cpu(slice_num * output_img_size);

    checkRuntime(cudaMemcpyAsync(input_image_.gpu(), image.bgrptr, size_image, cudaMemcpyHostToDevice, stream_));
    // checkRuntime(cudaStreamSynchronize(stream_));
    uint8_t* input_device = input_image_.gpu();
    uint8_t* output_device = output_images_.gpu();

    slice_plane(input_device, output_device, width, height, slice_width, slice_height, slice_num_h, slice_num_v, overlap_pixel, stream);

    checkRuntime(cudaStreamSynchronize(stream_));
    
    std::vector<SlicedImageData> slicedData(slice_num);
    for (int i = 0; i < slice_num; ++i) {
        slicedData[i].image = cv::Mat::zeros(slice_height, slice_width, CV_8UC3);
        slicedData[i].x = 0.0f;
        slicedData[i].y = 0.0f;
    }

    for (int i = 0; i < slice_num_h; i++)
    {
        int x = std::max(0, i * (slice_width - overlap_pixel));
        for (int j = 0; j < slice_num_v; j++)
        {
            int y = std::max(0, j * (slice_height - overlap_pixel));
            int index = i * slice_num_v + j;
            slicedData[index].x = x;
            slicedData[index].y = y;
            slicedData[index].w = slice_width;
            slicedData[index].h = slice_height;
            uint8_t* output_img_data = slicedData[index].image.ptr<uint8_t>();
            cudaMemcpyAsync(output_img_data, output_device+index*output_img_size, output_img_size*sizeof(uint8_t), cudaMemcpyDeviceToHost);
        }
    }
    checkRuntime(cudaStreamSynchronize(stream_));
    return slicedData;
}



}