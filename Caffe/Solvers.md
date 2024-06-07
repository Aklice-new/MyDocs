# Optimizer 优化器

        目前这些优化算法都是基于牛顿梯度下降法来做的。不同的优化器有不同的策略，来避免在梯度下降过程中的一些问题，如：跳不出局部最优解等。梯度下降的本质就是沿着梯度的反方向去更新参数，因为梯度的方向是目标函数值增大的方向。

        一般的梯度下降是指，给定待优化参数$x$, 和目标函数$\ell(x)$(在优化问题中我们习惯去定义$Loss$函数作为优化的目标)，学习率为$\eta$，在下降过程中一般会有下面几步：

1. 首先通过反向传播（backward的过程）求得当前待优化参数的梯度$\nabla \ell(x)$;

2. 设当前轮数为$i$，则更新完后的参数为$x_{i + 1} = x_{i} - \eta \nabla \ell(x_{i})$;

        不同的优化器，SGD、Momentum、 Nester、 Adagrad 、RMSprop 、 Adadelta、Adam 都是在此基础上采取不同的策略来使得整个优化过程更加合理。

## SGD

SGD(Stochastic Gradient Descent)：随机梯度下降法。SGD算法是最简单的方法，其更新过程就是：

1. 计算梯度当前时刻$t$的梯度$g_{t} = \nabla \ell(x_t)$;

2. 梯度下降，更新参数：$x_{t + 1} = x_{t} - \eta * g_{t}$;

它的缺点在于收敛的速度慢，由于固定的学习率，不能再具体的优化过程中很好的适应函数的变化，导致优化的速度慢。    

为了减缓这种问题的出现（即在极值两边进行震荡），引入了冲量（momentum）这个概念，它是用来描述梯度在下降过程中的惯性值。

一阶冲量的表达式为：$m_t = \beta m_{t - 1} + g_{t}$，其表达含义就是当前的冲量值与之前所有的梯度值都有关，

展开看： $m_t = g_{t} + \beta g_{t - 1} + \beta^2 g_{t - 2} + \beta^3 g_{t - 3} ... $，就是之前所有值的加权和。$\beta$常见的取值有0.5， 0.9， 0.95。

则更新方程为：$x_{t + 1} = x_{t} - \eta * m_{t + 1}$

<img src="file:///home/aklice/文档/MyDocs/Caffe/imgs/momentum_1.png" title="" alt="momentum_1.png" width="367"> 未加冲量优化的

<img title="" src="file:///home/aklice/文档/MyDocs/Caffe/imgs/momentum_2.png" alt="momentum_2.png" width="363" data-align="inline">加冲量优化的

实现：（以caffe1.0为例）在caffe sgd_solver.cpp中 ，关于梯度更新是分两步去做的：

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

## Nesterov（NAG）

在SGD with Momentum的基础上，既然我们已经知道了梯度会沿着冲量的方向下降这么多，为什么不直接在下降后的方向进行计算（相当于提前走到了梯度下降的位置上）。

所以在更新过程中计算公式就变成了：

$m_{t+1} = \beta*m_{t} + g(x_t - \eta*m_{t})$

$x_{t+1} = x_{t} + \eta * m_{t+1}$

在具体的实现过程中我们就会有疑问，上述过程中都这个$g(x_t - \eta * m_t)$这个梯度怎么求？求梯度的过程我们知道是先走一遍正向传播，再通过反向传播回来来求得参数的梯度，如果要这样更新的话，岂不是要再走一遍forward和backward的过程，显然这样太耗费时间了。所以在具体的实现过程中，需要对公式进行一些变换：

$x_{t} = x_{t} - \eta * m_{t}$

$m_{t+1} = \beta * m_{t} - \eta * g(x_t)$

$x_{t+1} = x_{t} + \beta^2*m_{t} - (1 + beta) * \eta * g(x_{t})$

在caffe中的实现部分在nesterov_solver.cpp中，具体的函数就是在ComputeUpdateValue()。以cuda代码为例：

```cpp
template <typename Dtype>
__global__ void NesterovUpdate(int N, Dtype* g, Dtype* h, Dtype momentum, Dtype local_rate)
{
    CUDA_KERNEL_LOOP(i, N)
    {
        float hi = h[i];
        float hi_new = h[i] = momentum * hi + local_rate * g[i];
        g[i] = (1 + momentum) * hi_new - momentum * hi;
// 这部分内容合并一下就是 \beta^2*m_{t} - (1 + beta) * \eta * g(x_{t}) 这部分内容
    }
}
```

## AdaGrad

上面所述的这些优化器都是以固定的学习率$\eta去更新待优化参数，而在深度学习的过程中，设计大量的不同的参数，不同的参数的更新的步长往往也不一致。对于更新不频繁的参数，我们希望更新的步长大写，学习率高些，反之亦然。

在此引入二阶动量：

$G_{t} = v_t = diag(\sum_{i=1}^{t}g^2_{i, 1},\sum_{i=1}^{t}g^2_{i, 2},...,\sum_{i=1}^{t}g^2_{i, d})$

在SGD的基础上，更新方程变为：

$x_{t+1, i} = x_{t, i} - \frac{\eta}{\sqrt{G_{t, ii} + \varepsilon }} * g_{t,i}$

其中$\varepsilon$是为了避免分母为0，是一个扰动项。

它的缺点是**分母会不断累积，学习率会变的非常小**。

代码实现：caffe这部分内容的实现在AdaGradUpdate中：

```cpp
// 其中h[i]保存的是第i个参数的累积梯度平方和
// g[i]是第i个参数的梯度
// 计算过程就如上述公式计算梯度
template <typename Dtype>
__global__ void AdaGradUpdate(int N, Dtype* g, Dtype* h, Dtype delta, Dtype local_rate)
{
    CUDA_KERNEL_LOOP(i, N)
    {
        float gi = g[i];
        float hi = h[i] = h[i] + gi * gi;
        g[i] = local_rate * gi / (sqrt(hi) + delta);
    }
}
```

## AdaDelta

AdaDelta是对AdaGrade的优化，因为梯度累积的原因，导致学习率会不断变小。所以将分母换成了过去的梯度的均方根root mean sqared(RMS)，所以更新梯度的方程变为：

$x_{t + 1, i} = x_{t, i} - \frac{\eta}{\sqrt{E[g_{i}^2]_{t}} + \varepsilon} * g_{t, i} = x_{t, i} - \frac{\eta}{RMS(g)_{t} + \varepsilon} * g_{t, i}$

其中E的计算公式为（这里在计算均值的时候直接通过这样加权的方式计算，避免存储历史值）：

$E[g^2]_{t} = \gamma * E[g^2]_{t - 1} + (1 - \gamma)g^2_{t}$ ,            $\gamma$常取0.9。

因为作者指出更新规则中必须与参数具有相同的假设单位，所以$\eta$被替换成为了梯度更新的RMS：

$E[\Delta x^2]_t = \gamma * E[\Delta x^2]_{t - 1} + (1 - \gamma) *E[\Delta x^2]_t$

则更新方程就变为了：

$x_{t + 1, i} = x_{t, i} - \frac{RMS(\Delta x^2)_{t - 1}}{RMS(g^2)_{t}} * g_{t}$    

对应到caffe中的实现，这部分内容在AdaDeltaUpdate中：

```cpp
template <typename Dtype>
// 参数列表中N为参数个数，
// g为需要计算的梯度，
// h为累积的梯度平方均值， 
// h2 为累积的梯度更新平方均值
// momentum为计算均方的因子
// delta是扰动项
// local_rate 学习率，其实adaDelta的更新方式已经不依赖于学习率了，不明白为什么还要在加一个
__global__ void AdaDeltaUpdate(int N, Dtype* g, Dtype* h, Dtype* h2, Dtype momentum, Dtype delta, Dtype local_rate)
{    
    CUDA_KERNEL_LOOP(i, N)
    {
        float gi = g[i];
        // 计算当前的梯度均方 hi
        float hi = h[i] = momentum * h[i] + (1 - momentum) * gi * gi;
        // 计算当前t时刻的梯度
        gi = gi * sqrt((h2[i] + delta) / (hi + delta));
        // 更新梯度变化均值 h2
        h2[i] = momentum * h2[i] + (1 - momentum) * gi * gi;
        // 计算当前参数具体更新的值
        g[i] = local_rate * gi;
    }
}
```

## RMSProp

其更新方式和AdaDelta的及其相似，建议其中$\gamma$取0.9，$\eta$取0.001。与之不同的地方在与RMSProp不需要计算梯度变化均值平方，所以更新的方程就是：

$x_{t + 1, i} = x_{t, i} - \frac{\eta}{\sqrt{E[g_{i}^2]*{t}} + \varepsilon} * g_{t, i} = x_{t, i} - \frac{\eta}{RMS(g)*{t} + \varepsilon} * g_{t, i}$

对应到caffe中的实现，这部分内容在RMSPropUpdate中：

```cpp
template <typename Dtype>
// 参数列表中N为参数个数，
// g为需要计算的梯度，
// h为累积的梯度平方均值，
// rms_decay为计算均值的因子
// delta是扰动项
__global__ void RMSPropUpdate(int N, Dtype* g, Dtype* h, Dtype rms_decay, Dtype delta, Dtype local_rate)
{
    CUDA_KERNEL_LOOP(i, N)
    {
        float gi = g[i];
        // 对应到AdaDelta中计算梯度均方的公式
        float hi = h[i] = rms_decay * h[i] + (1 - rms_decay) * gi * gi;
        // 计算梯度
        g[i] = local_rate * g[i] / (sqrt(hi) + delta);
    }
}
```

## Adam

自适应矩估计（Adaptive Moment Estimation），在AdaDelta和RMSProp的基础上，保存了梯度衰减平方均值，同时还保存了一个梯度指数衰减均值，类似于动量：

$m_t = \gamma * m_{t - 1} + (1 - \gamma)* g_t$

$v_t = \beta*v_{t - 1} + (1 - \beta)g_{t}^2$ 

实际上，$m_t, v_t$分别是对梯度的一阶矩和二阶矩估计。

展开$m_t$来看：

$m_t = (1 - \gamma)(g_t + \gamma*g_{t - 1} +\gamma^2*g_{t - 2}+ \gamma^3*g_{t - 3}+...)  $

因为$\sum_{i=0}^\infin \gamma^i = \frac{1}{1-\gamma}$，所以整体权重系数的和为1。

当t比较小的时候权重和就不等于0，而等于$\sum_{i=0}^t \gamma^i = \frac{1 - \gamma^t}{1-\gamma}$ 。

所以需要通过计算偏差矫正来抵消偏差。

$\hat{m_t} = \frac{m_t}{1 - \gamma^t}$  $\hat{v_t} = \frac{v_t}{1 - \beta^t}$

所以更新方程就变为了：

$x_{t + 1, i} = x_{t, i} - \frac{\eta*\hat{m_t}}{\sqrt{\hat{v_t}} + \varepsilon}$

作者建议β1取默认值为0.9，β2为0.999，ε​为1e-8。

目前在深度学习中用到的最多的还是

对应到caffe中的实现，这部分内容在RMSPropUpdate中：

```cpp

template <typename Dtype>
__global__ void AdamUpdate(int N, Dtype* g, Dtype* m, Dtype* v, Dtype beta1, Dtype beta2, Dtype eps_hat, Dtype corrected_local_rate){
    CUDA_KERNEL_LOOP(i, N)
    {
// 这里的代码也比较简单，就是计算两个矩估计，然后根据上述公式计算需要更新的梯度
        float gi = g[i];
        float mi = m[i] = m[i] * beta1 + gi * (1 - beta1);
        float vi = v[i] = v[i] * beta2 + gi * gi * (1 - beta2);
        g[i] = corrected_local_rate * mi / (sqrt(vi) + eps_hat);
    }
}
```

## 参考文章

[知乎# 从 SGD 到 Adam](https://zhuanlan.zhihu.com/p/32626442)

[Caffe源码-几种优化算法 - Rule110 - 博客园](https://www.cnblogs.com/Relu110/p/12115740.html#NAG)

[从 SGD 到 Adam —— 常见优化算法总结 - hejunlin - 博客园](https://www.cnblogs.com/hejunlin1992/p/13027288.html)

([11. 优化算法 &#8212; 动手学深度学习 2.0.0 documentation](https://zh.d2l.ai/chapter_optimization/index.html))
