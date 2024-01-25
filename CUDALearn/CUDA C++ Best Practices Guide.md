

# Chapter 12 性能度量

## 1. 计时器
在对CUDA程序计时时，使用CPU或者GPU的计时器都是可以的，下面是对这两种计时方法的分析：
### 1.1 使用CPU计时器

使用传统的基于CPU的计时器比如<chrono>等，但是在GPU上函数的执行是异步的，所以为了精确地测量调用CUDA函数的耗时，需要在启动和停止计时器时先试用cudaDeviceSynchronize()强制对CPU和GPU进行同步，cudaDeviceSynchronize()将调用的CPU线程堵塞，直到完成所有CUDA的操作。

