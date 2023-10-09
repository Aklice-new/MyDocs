

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