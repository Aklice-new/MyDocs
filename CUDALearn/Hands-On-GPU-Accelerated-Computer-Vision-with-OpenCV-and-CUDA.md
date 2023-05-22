# Learn CUDA

  

# Chapter 3

  

## kernel block thread

  

一个grid中有很多的block，然后block中又有很多的thread。grid可以是多维的，但是在计算thread的时候就需要注意了。如果grid是1维且每个block中只有一个thread，那么每次取threadid的时候就可以通过 blockIdx.x来取得。如果是二维的话，就可以 threadIdx.x + blockIdx.x * blockDim.x。具体的计算过程百度。还可以三维。<<<block的块数， 每个block中的线程数>>>，还有一点，在block中还有一个概念叫warp，每个warp中的线程是同步的，在设计的时候尽量线程数是32的倍数。

#### 关于 grid 和block的声明定义

一般定义grid和block的时候是通过dim3类型的变量来进行了，但是有时候直接使用了一维的所以直接给了一个int类型的值进去。
可以看看这篇博客中的图，方便理解。
https://blog.csdn.net/tiao_god/article/details/107181883。


  

## 原子操作

  

在cuda内的多线程操作的过程中，针对一些简单的加减操作等，可能会导致冲突导致计算错误，所以针对一些这样的操作可以进行原子操作（这里理解的不太清，具体什么情况会用到还没明白，代码中有个因没用原子操作而出现错误的例子）。

  

## global memeory shared memeory constant memeory

  

这三种内存是gpu编程中针对程序员可见可操作的内存区域，合理的设计安排内存的使用可以提高程序的运行效率。

- global memeory是最普通的全局内存区域，即申请的一般内存区域，grid内的线程都能共享访问的一片区域，同时访问该片区域的延迟最大。

- shared memeory是一段block内线程共享的区域,线程间可以通过这片区域通信，读写延迟相对于global memeory少，在声明过程中，可以直接确定空间大小，也可以在运行时决定空间大小。

- constant memeory针对host可以读写，但是对于设备来说只能进行访问，它的访问效率极高，但是只有64KB，同时每grid都有可以读的constant memeory。

  

### 全局内存

- 动态内存： 通过cudaMalloc/cudaFree进行申请和释放。

- 静态内存：声明 __device__ + 声明，如 __device int a;申请得到的变量可以在kernel中进行访问。设备上的变量主机不能直接访问，需要通过cudaMemcpyToSymbol/cudaMemcpyFromSymbol进行拷贝。定义的全局变量可以引用名字但是不能引用地址，需要使用getSymbolAddress进行地址转换（直接抄的，不明白）。

  

### 共享内存

  

- 一个block中的线程是共享这片区域的

- 数据量较小直接访问全局内存比shared memory快shared memory

- 加速大量数据访存的原理是硬件上支持多bank并发访存，需要避免bank conflict

  

#### 空间申请

  

- 静态申请 __shared__ float a[10][10];

- 动态申请 extern float a; 然后再启动核函数时加入空间 kernel<<<grid,block,size*sizeof(int)>>>()；（没用过，等会试试）

  

#### 线程同步

  

- 障碍：所有调用线程等待其余调用线程达到障碍点 。 _syncthreads(); 注意，要保证所有线程都能到达这里。

- 内存栅栏：所有调用内存必须等待内存修改对其余线程可见才能继续进行。没试过。

  

### 静态内存

通过 __constant__ 关键词修饰。如 __constant__ int a;然后通过 cudaMemcpyToSymbol/cudaMemcpyFromSymbol进行拷贝。

  

### 纹理内存

  

可以定义1D / 2D / 3D三种方式的纹理内存，纹理内存时只读内存，如果在当前线程访问过程中可能会同时访问临近区域或线程的数据，可以使用该内存来提高访问效率，如图像滤波。

通过 texture<float, dim(1,2,3), cudaReadModeElementType> name;


# Chapter4

## CUDA Event

使用Cuda Event来对cuda的性能进行测试，使用普通的cpu的计时方法也是可以的，但是cudaEvent是在GPU上进行计时的，所以理论上更准一些。这里这篇博客不错[https://blog.csdn.net/litdaguang/article/details/77585011]。

cudaEvent_t 声明一个事件。
cudaEventCreate 
cudaEventRecord 进行记录

## CUDA Error Handling

书中所讲的是通过cudaError_t cudaStatus;来获得每条CUDA语句的执行结果。然后还可以通过sample文件夹中的helper_cuda.h中的错误检测来进行。这里有篇博客[https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api]

