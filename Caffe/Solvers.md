# Optimizer 优化器

        目前这些优化算法都是基于牛顿梯度下降法来做的。不同的优化器有不同的策略，来避免在梯度下降过程中的一些问题，如：跳不出局部最优解等。梯度下降的本质就是沿着梯度的反方向去更新参数，因为梯度的方向是目标函数值增大的方向。

        一般的梯度下降是指，给定待优化参数$x$, 和目标函数$\ell(x)$(在优化问题中我们习惯去定义$Loss$函数作为优化的目标)，学习率为$\eta$，在下降过程中一般会有下面几步：

1. 首先通过反向传播（backward的过程）求得当前待优化参数的梯度$\nabla \ell(x)$;

2. 设当前轮数为$i$，则更新完后的参数为$x_{i + 1} = x_{i} - \eta \nabla \ell(x_{i})$;

        不同的优化器，SGD、Momentum、 Nester、 Adagrad 、RMSprop 、 Adadelta、Adam 都是在此基础上采取不同的策略来使得整个优化过程更加合理。

## SGDSolver

SGD(Stochastic Gradient Descent)：随机梯度下降法。SGD算法是最简单的方法，其更新过程就是：

1. 计算梯度$\nabla x$;

2. 梯度下降，更新参数：$x_{i + 1} = x_{i} - \eta \nabla x_{i}$;

它的缺点在于收敛的速度慢，由于固定的学习率，不能再具体的优化过程中很好的适应函数的变化，导致优化的速度慢。    

为了减缓这种问题的出现（即在极值两边进行震荡），引入了冲量（momentum）这个概念，它是用来描述梯度在下降过程中的惯性值。

一阶冲量的表达式为：$m_i = \beta m_{i - 1} + \nabla \ell(x_i)$，其表达含义就是当前的冲量值与之前所有的梯度值都有关，

展开看： $m_i =\nabla \ell(x_i) + \beta \nabla \ell_{i - 1} + \beta^2 \nabla \ell_{i - 2} + \beta^3 \nabla \ell_{i - 3} ... $，就是之前所有值的加权和。$\beta$常见的取值有0.5， 0.9， 0.95。

则更新方程为：$x_{i + 1} = x_{i} - \eta * m_{i + 1}$

<img src="file:///home/aklice/文档/MyDocs/Caffe/imgs/momentum_1.png" title="" alt="momentum_1.png" width="367"> 未加冲量优化的

<img title="" src="file:///home/aklice/文档/MyDocs/Caffe/imgs/momentum_2.png" alt="momentum_2.png" width="363" data-align="inline">加冲量优化的

实现：（以caffe1.0为例）在caffe中，关于梯度更新是分两步去做的：

首先是一个solver->ComputeUpdateValue()这个函数是计算梯度下降的具体值。

然后net->Update()再进行对待优化参数的更新。

```cpp
template <typename Dtype>
void SGDSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate)
{    
    // 获取一下当前时刻i的参数信息
    const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
    // 获取参数学习率
    const vector<float>& net_params_lr = this->net_->params_lr();
    // 获取参数的历史冲量
    Dtype momentum = this->param_.momentum();
    // 计算当前学习率
    Dtype local_rate = rate * net_params_lr[param_id];
    // Compute the update to history, then copy it to the parameter diff.
    // 下面这部分计算内容就是梯度下降，同时保存历史冲量
    // cpu部分使用 mkl库进行加速
    // gpu部分使用cuda加速
    switch (Caffe::mode())
    {
    case Caffe::CPU:
    {
        caffe_cpu_axpby(net_params[param_id]->count(), local_rate, net_params[param_id]->cpu_diff(), momentum,
            history_[param_id]->mutable_cpu_data());
        caffe_copy(
            net_params[param_id]->count(), history_[param_id]->cpu_data(), net_params[param_id]->mutable_cpu_diff());
        break;
    }
    case Caffe::GPU:
    {
#ifndef CPU_ONLY
        sgd_update_gpu(net_params[param_id]->count(), net_params[param_id]->mutable_gpu_diff(),
            history_[param_id]->mutable_gpu_data(), momentum, local_rate);
#else
        NO_GPU;
#endif
        break;
    }
    default: LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
    }
}
## NesterSolver
```

cuda加速部分可以简单介绍一下：

```cpp
// 这里CUDA_KERNEL_LOOP这个宏就是计算线程id的一个宏
// 这是一个一维的block，将要计算更新的长度为n的待优化参数划分成了blockDim.x个block
// 为什么不让一个线程计算连续的blockDim.x个待优化参数？答：为了合并访存
// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

template <typename Dtype>
__global__ void SGDUpdate(int N, Dtype* g, Dtype* h,
    Dtype momentum, Dtype local_rate) {
  CUDA_KERNEL_LOOP(i, N) {
// 这里就是更新的核心部分，更新h(历史冲量记录)和g(这里的g计算的是)
    g[i] = h[i] = momentum*h[i] + local_rate*g[i];
  }
}
template <typename Dtype>
void sgd_update_gpu(int N, Dtype* g, Dtype* h, Dtype momentum,
    Dtype local_rate) {
  SGDUpdate<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, g, h, momentum, local_rate);
  CUDA_POST_KERNEL_CHECK;
}
```

在net->Update()里面将$x = x - g$这样的操作看成是向量的乘法同样通过第三方库去做的加速。

## NesterovSolver （NGD）

在SGD with Momentum的基础上，

## AdaGradSolver

## RMSPropSolver

## AdaDeltaSolver

## AdamSolver

## 参考文章

[知乎# 从 SGD 到 Adam](https://zhuanlan.zhihu.com/p/32626442)
