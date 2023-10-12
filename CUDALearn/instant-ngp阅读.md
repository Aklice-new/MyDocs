
## 一些有用的第三方库的简单学习
# tinny-cuda-nn

1. 封装起来的管理GPU内存的工具 gpu_memory.h
```cpp
//提供了众多在GPU内存管理上的函数，如：
void allocate_memory(size_t n_bytes);
void free_memory();
void resize(const size_t size); 
//等等，许多有用的函数接口
```
2. 在启动核函数的时候，使用统一的模版接口启动：
```cpp

//这个是统一的模版接口
template <typename K, typename T, typename ... Types>

inline void linear_kernel(K kernel, uint32_t shmem_size, cudaStream_t stream, T n_elements, Types ... args) {

if (n_elements <= 0) {

return;

}

kernel<<<n_blocks_linear(n_elements), N_THREADS_LINEAR, shmem_size, stream>>>(n_elements, args...);

}

//对于任意一个核函数
template <typename T>

__global__ void to_ldr(const uint64_t num_elements, const uint32_t n_channels, const uint32_t stride, const T* __restrict__ in, uint8_t* __restrict__ out) {

const uint64_t i = threadIdx.x + blockIdx.x * blockDim.x;

if (i >= num_elements) return;

  

const uint64_t pixel = i / n_channels;

const uint32_t channel = i - pixel * n_channels;

  

out[i] = (uint8_t)(powf(fmaxf(fminf(in[pixel * stride + channel], 1.0f), 0.0f), 1.0f/2.2f) * 255.0f + 0.5f);

}
// 启动的时候使用的是
linear_kernel(to_ldr<T>, 0, nullptr, width * height * n_channels, n_channels, channel_stride, image, image_ldr.data());
```
3. 使用cuda封装好的内存对象：cudaResourceDesc、cudaTextureDesc、cudaTextureObject_t ，。
```cpp
cudaResourceDesc resDesc;

memset(&resDesc, 0, sizeof(resDesc));

resDesc.resType = cudaResourceTypePitch2D;

resDesc.res.pitch2D.devPtr = image.data();

resDesc.res.pitch2D.desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);

resDesc.res.pitch2D.width = width;

resDesc.res.pitch2D.height = height;

resDesc.res.pitch2D.pitchInBytes = width * 4 * sizeof(float);

  

cudaTextureDesc texDesc;

memset(&texDesc, 0, sizeof(texDesc));

texDesc.filterMode = cudaFilterModeLinear;

texDesc.normalizedCoords = true;

texDesc.addressMode[0] = cudaAddressModeClamp;

texDesc.addressMode[1] = cudaAddressModeClamp;

texDesc.addressMode[2] = cudaAddressModeClamp;

  

cudaTextureObject_t texture;

CUDA_CHECK_THROW(cudaCreateTextureObject(&texture, &resDesc, &texDesc, nullptr));
```

## fmt  开源的格式化输出的库

可以很方便的替代cout这个垃圾玩意。
关于具体的使用example不在赘述，[https://github.com/fmtlib/fmt] 这个主页有一些说明。
需要注意的是在使用过程中，fmt是可以header only的，但是需要在使用过程中加入宏定义说明
``` cpp

#define FMT_HEADER_ONLY // 注意就是这个宏定义。不然会报错。
#include <fmt/format.h>

```