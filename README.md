# CUDA-SAHI-CROP-IMAGE
## 功能
使用cuda实现类似sahi库的切图功能, 返回切割后的图片和相对于原图的坐标

## 说明
按照sahi接口进行了改造, 入口函数提供子图大小, 宽高重合比率。

## 增加autoSlice函数
会根据图片的宽高自动切分成子图，按照sahi源码改的

## 优化核函数
### 核函数设计
```C++
static __global__ void slice_kernel(
  const uchar3* __restrict__ image,
  uchar3* __restrict__ outs,
  const int width,
  const int height,
  const int slice_width,
  const int slice_height,
  const int slice_num_h,
  const int slice_num_v,
  const int* __restrict__ slice_start_point)
{
    const int slice_idx = blockIdx.z;
    const int start_x = slice_start_point[slice_idx * 2];
    const int start_y = slice_start_point[slice_idx * 2 + 1];

    // 当前像素在切片内的相对位置
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= slice_width| y >= slice_height) 
    {
        return;
    }
        
    // 原图坐标
    const int dx = start_x + x;
    const int dy = start_y + y;

    if(dx >= width || dy >= height) 
        return;

    // 读取像素
    const int src_index = dy * width + dx;
    const uchar3 pixel = image[src_index];

    // 写入切片
    const int dst_index = slice_idx * slice_width * slice_height + y * slice_width + x;
    outs[dst_index] = pixel;
}
```
### 说明
- 三维网格布局：blockIdx.z表示切片索引。消去了之前核函数中的for循环。
- 优化坐标计算：只保留起始点坐标，代码更易读

### 展示效果
![效果](https://github.com/leon0514/cuda-sahi-crop-image/blob/main/workspace/test.gif)