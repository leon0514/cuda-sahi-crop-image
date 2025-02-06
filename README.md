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
__global__ void slice_kernel(
  const uchar3* image,       // 输入图像
  uchar3* outs,              // 输出切片
  const int width,           // 原图宽度
  /* 其他参数... */
){
    // 计算切片索引和坐标
    const int slice_idx = blockIdx.z;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // 边界检查
    if(x >= slice_width || y >= slice_height) return;
    
    // 坐标映射与数据拷贝
    const int dx = start_x + x;
    const int dy = start_y + y;
    if(dx < width && dy < height){
        outs[dst_index] = image[src_index];
    }
}
```
### 说明
- 三维网格布局：blockIdx.z表示切片索引。消去了之前核函数中的for循环。
- 优化坐标计算：只保留起始点坐标，代码更易读

### 展示效果
![效果](https://github.com/leon0514/cuda-sahi-crop-image/blob/main/workspace/test.gif)