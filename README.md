# CUDA-SAHI-CROP-IMAGE
## 功能
使用cuda实现类似sahi库的切图功能, 返回切割后的图片和相对于原图的坐标

## 说明

**限制切割16x16份图，代码里面使用共享内存存储的切割起始点。**
```C++
// 定义共享内存，存储切片范围
__shared__ int slice_range_h[32]; // 假设一个线程块最多处理 16 个切片（可以根据实际情况调整）
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
```
重叠面积按照最短边乘重叠百分比计算。
```C++
int width = image.width;
int height = image.height;

int overlap_pixel = std::min(width, height) * overlap_ratio;
```

### 展示效果
![效果](https://github.com/leon0514/cuda-sahi-crop-image/blob/main/workspace/test.gif)