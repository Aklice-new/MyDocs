

# Chapter 12 性能度量 Performance Metrics

## 12.1 计时器
在对CUDA程序计时时，使用CPU或者GPU的计时器都是可以的，下面是对这两种计时方法的分析：
### 12.1.1 使用CPU计时器

使用传统的基于CPU的计时器比如<chrono>等，但是在GPU上函数的执行是异步的，所以为了精确地测量调用CUDA函数的耗时，需要在启动和停止计时器时先试用cudaDeviceSynchronize()强制对CPU和GPU进行同步，cudaDeviceSynchronize()将调用的CPU线程堵塞，直到完成所有CUDA的操作。
虽然在GPU上也可以将CPU线程与特定的流或事件进行同步，但是这些同步函数并不适用于默认流以外的流中的计时器，cudaStreamSynchronize ( )将CPU线程阻塞，直到之前发送到给定流中的所有CUDA调用都完成。cudaEventSynchronize ( )阻塞，直到特定流中的给定事件被GPU记录。由于驱动程序可能会交织执行来自其他非默认流的CUDA调用，因此可能会将其他流中的调用包含在计时中。但是默认流Stream0在device上工作表现出串行性，所以可以直接使用这些函数进行计时。
最后需要注意一点的是，CPU和GPU的同步操作会强制阻塞，所以会最大限度的降低性能。
### 12.1.2 使用GPU计时器

```cpp
cudaEvent_t start, stop; 
float time; 
cudaEventCreate(&start); 
cudaEventCreate(&stop); 
cudaEventRecord( start, 0 );  // 这里的0是指 Stream0，也就是默认流
kernel<<<grid,threads>>> ( d_odata, d_idata, size_x, size_y, NUM_REPS); 
cudaEventRecord( stop, 0 );  // 同样这里也是
cudaEventSynchronize( stop ); 
cudaEventElapsedTime( &time, start, stop ); 
cudaEventDestroy( start ); 
cudaEventDestroy( stop );

```
使用cudaEventRecord将start 和 stop这两个事件放入到Stream0中，当事件到达Stream0时，会记录时间戳，使用cudaEventSynchronize对CPU和GPU进行同步，然后使用cudaEventElapsedTime进行计时。这种计时时使用GPU时钟进行计时的，

## 12.2 带宽 (Band Width) **Hight Priority**

带宽衡量数据的传输速率，在性能优化中是很重要的一环，为了准确测量性能，需要计算理论带宽和有效带宽。当后者远低于前者时，设计或实现细节都有可能降低带宽，提高带宽应该是后续优化工作的首要目标。

### 12.2.1 理论带宽计算

可以根据显卡的硬件规格参数进行计算。
例如，NVIDIA Tesla V100采用HBM2 (双倍数据率) RAM，其存储时钟速率为877 MHz，具有4096位宽的存储器接口。
$$(0.877 \times 10^9 \times 4096/8 \times 2) \div 10^9 = 898GB/s$$
### 12.2.2 有效带宽计算

有效带宽用于了解在程序运行过程中数据数如何访问的，下面是计算公式
$$Effective Band Width = ((Byte_{read} + Byte_{write}) \div 10^9) \div times$$
(通常使用GB/s作为单位)
这里Byte_read 和Byte_wirte分别是在kernel中读取和写入的字节数。
例如，计算一个对$2048 \times 2048$的矩阵进行拷贝的操作的带宽：
$$((2048 ^ 2 \div 4 \times 2) \div 10^9) \div time$$
这里的4是float类型的字节，2是包含了读写两次操作。

### 12.2.3 Visual Profiler
使用Visual Profiler可以查看一下几个吞吐量指标：
- Requested Global Load Throughput  
- Requested Global Store Throughput
- Global Load Throughput 
- Global Store Throughput 
- DRAM Read Throughput 
- DRAM Write Throughput

其中 Requested Global Load Throughput 和 Request Global Store Throughput表示的是kernel中的全局内存吞吐量，用于计算Effective Band Width有用。
而在程序运行过程中整体的吞吐量还会计算一些不会使用的数据传输，也就是Global Load Throughput 和 Global Load Throughput。
这两者的比较可以有效的看出Global Memory被浪费了多少资源，对于Global Memory访问，这种请求的内存带宽与实际的内存带宽的比较是通过Global Memory Load Efficiency和Global Memory Store Efficiency的度量来报告的。


# Chapter13 内存优化 Memory Optimizations 

内存优化是性能优化中最重要的领域。其目标是通过最大化带宽来最大化硬件的使用。带宽是通过使用尽可能多的快速存储器和尽可能少的慢速存储器来实现的。本章讨论了主机和设备上的各种内存，以及如何更好地设置数据项来有效地使用内存。

## 13.1 Host的Device的数据传输

首先，device memory和GPU之间的带宽（上文中提到的V100 898GB/s）远高于host memory和device memory之间的带宽（16GB/s  pcie x16 Gen3）。

