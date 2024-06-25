## Caffe中的Layer类

在caffe中将对数据的处理步骤抽象成为层这样的概念，它不用去关心数据的存储，而只关心数据在整个框架中的处理过程，根据不同的处理类型对层进行了划分：

- Data Layers：处于整个网络结构的最低端，完成了数据的读入和写出过程。

- Vision Layers：一般用于对图像数据的处理，输入位图像数据，输出也是图像数据。一般的操作有卷积、池化 、裁剪等操作。

- Recurrent Layers：循环层，包含一些常见的循环神经网络的实现，如：RNN、LSTM。

- Activation Layers：定义了一些逐元素的操作和一些激活函数。

- Utility Layers：定义了特殊功能的层。

- Loss Layers：定义了计算loss的一些层。

- Common Layers：常见的一些网络中的层，全连接层、dropout、Embed。

- Normalization Layers: 用于正则化的一些层，包括LRN、MVN、BN。

关于Layer类中执行流程的各个步骤在这篇博客中有提到 [Caffe源码理解3：Layer基类与template method设计模式 - shine-lee - 博客园](https://www.cnblogs.com/shine-lee/p/10144341.html)

## 2. Vision Layers

im2col

sparse matrix 

### 2.1 Convolution

convolution

deconvolution https://www.zhihu.com/question/48279880

### 2.2 Pooling

maxpooling

avgpooling           

### 2.3 full-connected (inner_product)

## 3. Neuron Layers

这一层主要包含的的是对数据的单目操作的一些运算，包括取绝对值等一些常见的操作。

### 3.1 absval

forward:

$Z^{[l]} = |A^{[l-1]}|$

backward: 

$\frac{\partial Loss}{\partial A^{[l-1]}}  = \left\{\begin{aligned} \partial Z^{[l]}, A^{[l - 1]}\ge0; \\-\partial Z^{[l]}, A^{[l - 1]}\lt0 \end{aligned} \right.$

在caffe中这部分内容的实现在 absval_layer.cpp。

```cpp
template <typename Dtype>
void AbsValLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    const int count = top[0]->count();
    Dtype* top_data = top[0]->mutable_cpu_data();
    // 正向传播过程， count为数据的数量，bottom[0]为输入， top_data为输出
    caffe_abs(count, bottom[0]->cpu_data(), top_data);
}

template <typename Dtype>
void AbsValLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
    const int count = top[0]->count();
    const Dtype* top_diff = top[0]->cpu_diff();
    if (propagate_down[0])
    {
        const Dtype* bottom_data = bottom[0]->cpu_data();
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
        // 反向传播的主要过程，根据bottom_data的正负性对bottom_diff的正负做出改变
        caffe_cpu_sign(count, bottom_data, bottom_diff);
        // 将top的梯度传递过来
        caffe_mul(count, bottom_diff, top_diff, bottom_diff);
    }
}
```

### 3.2 bnll（二项正态对数似然）

forward:

$Z^{[l]} = \left\{\begin{aligned} A^{[l - 1]} +  log(1 + exp(-A^{[l - 1]}) = log(1 + exp(A^{[l - 1]}), if( A^{[l - 1]}\ge0) \\ log(1 + exp(A^{[l - 1]})), otherwise \end{aligned} \right.$    

backward:

$\frac{\partial Loss}{\partial A^{[l - 1]}} = \partial Z^{[l]} *\frac{exp(A^{[l - 1]})}{1 + exp(A^{[l - 1]})}$ 

为什么要在大于0的时候做这样的处理，是因为在指数的增长是非常快的，为了避免溢出。

```cpp
const float kBNLL_THRESHOLD = 50.;

template <typename Dtype>
void BNLLLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    const int count = bottom[0]->count();
    for (int i = 0; i < count; ++i)
    {    

        // 正向传播的过程， 大于0的时候做一个小的变换
        top_data[i]
            = bottom_data[i] > 0 ? bottom_data[i] + log(1. + exp(-bottom_data[i])) : log(1. + exp(bottom_data[i]));
    }
}

template <typename Dtype>
void BNLLLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
    if (propagate_down[0])
    {
        const Dtype* bottom_data = bottom[0]->cpu_data();
        const Dtype* top_diff = top[0]->cpu_diff();
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
        const int count = bottom[0]->count();
        Dtype expval;
        for (int i = 0; i < count; ++i)
        {
            // 反向传播的计算，这里为了避免溢出，同样也加了一个kBNLL_THRESHOLD的限制。
            expval = exp(std::min(bottom_data[i], Dtype(kBNLL_THRESHOLD)));
            bottom_diff[i] = top_diff[i] * expval / (expval + 1.);
        }
    }
}
```

### 3.3 dropout

        dropout操作是一种在深度学习中为了避免过拟合而提出来的一种操作，基本思路就是在**训练阶段**随机的去丢弃一些神经元，这些神经元不参与前向传播和反向传播的过程，这种操作可以强制网络学习到更鲁棒的特征表达,防止过度依赖某些特定的神经元组合,从而提高网络的泛化能力。

        一般来说，会有一个保留率$\theta 通常为0.5$, 同时还有一个$scale$调整因子 $scale = \frac{1}{\theta}$  。这个调整因子是为了保证在训练和测试时loss的稳定性，保证$E[dropout(X)] = X$， 因为在训练过程中会丢弃一部分神经元，这部分没有计算loss，所以需要加入scale来进行调整。

forward:

$Z^{[l]} = A^{[l - 1]} * mask * scale$   , $mask$是通过正态分布根据$\theta$随机得到的需要保留或者丢弃的神经元。

backward:

$\frac{\partial Loss}{\partial A^{[l - 1]}} = \partial Z * mask * scale$ 

caffe这部分内容的实现也比较简单，需要注意的是dropout层只在训练阶段会用到，其他阶段并不会用到。

```cpp
template <typename Dtype>
void DropoutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    unsigned int* mask = rand_vec_.mutable_cpu_data();
    const int count = bottom[0]->count();
    if (this->phase_ == TRAIN)
    {
        // Create random numbers
        caffe_rng_bernoulli(count, 1. - threshold_, mask);
        for (int i = 0; i < count; ++i)
        {    
            // 正向传播过程
            top_data[i] = bottom_data[i] * mask[i] * scale_;
        }
    }
    else
    {   // 测试阶段不需要进行dropout
        caffe_copy(bottom[0]->count(), bottom_data, top_data);
    }
}

template <typename Dtype>
void DropoutLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
    if (propagate_down[0])
    {
        const Dtype* top_diff = top[0]->cpu_diff();
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
        if (this->phase_ == TRAIN)
        {
            const unsigned int* mask = rand_vec_.cpu_data();
            const int count = bottom[0]->count();
            for (int i = 0; i < count; ++i)
            {    
                // 反向传播过程        
                bottom_diff[i] = top_diff[i] * mask[i] * scale_;
            }
        }
        else
        {   //  测试阶段计算梯度，直接将梯度回传
            caffe_copy(top[0]->count(), top_diff, bottom_diff);
        }
    }
}
```

### 3.4 elu

1. 非线性性更强，ELU可以学习到更加复杂的非线性映射,相比ReLU等线性修正单元具有更强的表达能力。

2. 输出均值接近0， ELU的输出均值往往更接近0,这有利于梯度的传播,提高了收敛速度。

3. 抑制死亡神经元，对于负值输入,ReLU会产生"死亡"的神经元,而ELU则可以通过指数函数缓慢恢复,避免了这一问题。

4. 抗噪性强，ELU对于输入噪声的鲁棒性更强,可以更好地处理输入数据中的噪声。

5. 收敛更快，由于ELU的输出均值接近0,以及抑制了死亡神经元的问题,相比ReLU等激活函数,ELU通常能够训练得更快。

6. 泛化性能好，由于上述特点,ELU通常能够学习到更好的特征表示,从而在各种任务上表现更出色。

forward:

$Z^{[l]} = \left\{\begin{array}{lr}A^{[l - 1]} & \mathrm{if} \; A^{[l - 1]} > 0 \\\alpha (\exp(A^{[l - 1]})-1) & \mathrm{if} \; A^{[l - 1]} \le 0 \end{array} \right.$    

backward：

$\frac{\partial Loss}{\partial A^{[l - 1]}} = \left\{ \begin{aligned} \partial Z^{[l]} ; if A^{[l - 1]} > 0 \\ \partial Z^{[l]} * \alpha * \exp(A^{[l - 1]}) ; if A^{[l - 1]} \le 0  \end{aligned} \right.$

在caffe的实现中：

```cpp
template <typename Dtype>
void ELULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    const int count = bottom[0]->count();
    Dtype alpha = this->layer_param_.elu_param().alpha();
    for (int i = 0; i < count; ++i)
    {    

        // 正向传播的过程
        top_data[i] = std::max(bottom_data[i], Dtype(0)) + alpha * (exp(std::min(bottom_data[i], Dtype(0))) - Dtype(1));
    }
}

template <typename Dtype>
void ELULayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
    if (propagate_down[0])
    {
        const Dtype* bottom_data = bottom[0]->cpu_data();
        const Dtype* top_data = top[0]->cpu_data();
        const Dtype* top_diff = top[0]->cpu_diff();
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
        const int count = bottom[0]->count();
        Dtype alpha = this->layer_param_.elu_param().alpha();
        for (int i = 0; i < count; ++i)
        {   // 这里是反向传播的过程，这里有一个小的trick
            // 在计算$\alpha * \exp(A^{[l - 1]})$的时候直接使用了正向传播的结果 top_data[i] + alpha来求得该值
            // top_data[i] = alpha * exp(A^{[l - 1]}) - alpha
            // 节省了一次计算exp
            bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0) + (alpha + top_data[i]) * (bottom_data[i] <= 0));
        }
    }
}
```

### 3.5 exp

forward:

$ Z^{[l]} = \gamma ^ {\alpha A^{[l - 1]} + \beta} $  其中，$\gamma$不指定的话默认是自然底数$\exp$ 

backward:

$\frac{\partial Loss}{\partial A^{[l - 1]}} = \partial Z * Z^{[l]} * \alpha * \log_e(\gamma)$  

```cpp
template <typename Dtype>
void ExpLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    const int count = bottom[0]->count();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    if (inner_scale_ == Dtype(1))
    {
        caffe_exp(count, bottom_data, top_data);
    }
    else
    {
        caffe_cpu_scale(count, inner_scale_, bottom_data, top_data);
        caffe_exp(count, top_data, top_data);
    }
    if (outer_scale_ != Dtype(1))
    {
        caffe_scal(count, outer_scale_, top_data);
    }
}

template <typename Dtype>
void ExpLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
    if (!propagate_down[0])
    {
        return;
    }
    const int count = bottom[0]->count();
    const Dtype* top_data = top[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_mul(count, top_data, top_diff, bottom_diff);
    if (inner_scale_ != Dtype(1))
    {
        caffe_scal(count, inner_scale_, bottom_diff);
    }
}
```

### 3.6 power

forward:

$Z^{[l]} = (\alpha * A^{[l - 1]} + \beta)^\gamma$

backward:

$\frac{\partial Loss}{\partial A^{[l - 1]}} = \gamma * \alpha * (\alpha * A^{[l - 1]} + \beta)^{\gamma - 1}$ 

$\frac{\partial Loss}{\partial A^{[l - 1]}} = \frac{\partial Loss}{\partial A^{[l - 1]}}\alpha \gamma (\alpha A^{[l-  1]} + \beta) ^ {\gamma - 1} =\frac{\partial E}{\partial Z^{[l]}}\frac{\alpha \gamma Z^{[l]}}{\alpha A^{[l- 1]} + \beta}$    

```cpp
// Compute y = (shift + scale * x)^power
template <typename Dtype>
void PowerLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    Dtype* top_data = top[0]->mutable_cpu_data();
    const int count = bottom[0]->count();
    // Special case where we can ignore the input: scale or power is 0.
    if (diff_scale_ == Dtype(0))
    {
        Dtype value = (power_ == 0) ? Dtype(1) : pow(shift_, power_);
        caffe_set(count, value, top_data);
        return;
    }
    const Dtype* bottom_data = bottom[0]->cpu_data();
    caffe_copy(count, bottom_data, top_data);
    if (scale_ != Dtype(1))
    {
        caffe_scal(count, scale_, top_data);
    }
    if (shift_ != Dtype(0))
    {
        caffe_add_scalar(count, shift_, top_data);
    }
    if (power_ != Dtype(1))
    {
        caffe_powx(count, top_data, power_, top_data);
    }
}

template <typename Dtype>
void PowerLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
    if (propagate_down[0])
    {
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
        const int count = bottom[0]->count();
        const Dtype* top_diff = top[0]->cpu_diff();
        if (diff_scale_ == Dtype(0) || power_ == Dtype(1))
        {
            caffe_set(count, diff_scale_, bottom_diff);
        }
        else
        {
            const Dtype* bottom_data = bottom[0]->cpu_data();
            // Compute dy/dx = scale * power * (shift + scale * x)^(power - 1)
            //               = diff_scale * y / (shift + scale * x)
            if (power_ == Dtype(2))
            {
                // Special case for y = (shift + scale * x)^2
                //     -> dy/dx = 2 * scale * (shift + scale * x)
                //              = diff_scale * shift + diff_scale * scale * x
                caffe_cpu_axpby(count, diff_scale_ * scale_, bottom_data, Dtype(0), bottom_diff);
                if (shift_ != Dtype(0))
                {
                    caffe_add_scalar(count, diff_scale_ * shift_, bottom_diff);
                }
            }
            else if (shift_ == Dtype(0))
            {
                // Special case for y = (scale * x)^power
                //     -> dy/dx = scale * power * (scale * x)^(power - 1)
                //              = scale * power * (scale * x)^power * (scale * x)^(-1)
                //              = power * y / x
                const Dtype* top_data = top[0]->cpu_data();
                caffe_div(count, top_data, bottom_data, bottom_diff);
                caffe_scal(count, power_, bottom_diff);
            }
            else
            {
                caffe_copy(count, bottom_data, bottom_diff);
                if (scale_ != Dtype(1))
                {
                    caffe_scal(count, scale_, bottom_diff);
                }
                if (shift_ != Dtype(0))
                {
                    caffe_add_scalar(count, shift_, bottom_diff);
                }
                const Dtype* top_data = top[0]->cpu_data();
                caffe_div<Dtype>(count, top_data, bottom_diff, bottom_diff);
                if (diff_scale_ != Dtype(1))
                {
                    caffe_scal(count, diff_scale_, bottom_diff);
                }
            }
        }
        if (diff_scale_ != Dtype(0))
        {
            caffe_mul(count, top_diff, bottom_diff, bottom_diff);
        }
    }
}
```

### 3.7 log

forward:

$Z^{[l]} = \log_{\gamma}(\alpha * A^{[l - 1]} + \beta)$

backward:

$\frac{\partial Loss }{\partial A^{[l - 1]}} = \partial Z^{[l]} * \frac{\alpha}{(\alpha * A^{[l - 1]} + \beta) * \ln(\gamma)}$ 

$\frac{\partial E}{\partial x} =\frac{\partial E}{\partial y} y \alpha \log_e(\gamma)$

### 3.8 prelu

forward:

$Z^{[l]} = \max(0, A^{[l - 1]}) +  * \min(0, A^{[l - 1]})$

$$

### 3.9 relu

forward:

$Z^{[l]} = \max(A^{[l - 1]}, 0)$

backward:

$\frac{\partial Loss}{\partial A^{[l - 1]}} = \left\{\begin{aligned} \partial Z ; if A^{[l - 1]} \gt0\\ 0; else\end{aligned} \right.$    

### 3.10 sigmod

forward:

$Z^{[l]} = (1 + \exp(-A^{[l - 1]})^{-1}$

backward:

$\frac{\partial Loss }{\partial A^{[l - 1]}} = \partial Z * \frac{\exp(-A^{[l - 1]})}{(1 + \exp(-A^{[l - 1]})^{2}} = \partial Z * Z^{[l]} * (1 - Z^{[l]})$ 

  

### 3.11 tanh

forward:

$Z^{[l]} = \frac{\exp(2*A^{[l - 1]}) - 1}{\exp(2*A^{[l - 1]}) + 1}$

backward:

$\frac{\partial Loss}{\partial A^{[l - 1]}} = \partial Z * \left(1 - \left[\frac{\exp(2*A^{[l - 1]}) - 1}{exp(2*A{[l - 1]}) + 1} \right]^2 \right) = \partial Z * (1 - {Z^{[l]}}^2)$

### 3.12 threshold

forward:

$Z^{[l]} = \left\{ \begin{array}{lr} 0 &\mathrm{if}\; A^{[l - 1]}\le t \\ 1 &\mathrm{if}\; A^{[l - 1]}\gt t \end{array}\right.$    

不连续，不可导

## 4. Loss Layers

### 4.1 multinomial_logistic_loss_layer

交叉熵损失，适用于多分类问题，同时易于优化。

forward:(其中N为样本个数，K为类别个数)

$Loss = -\sum_{i = 0}^{N}\sum_{k = 0}^{K} y_k* \log(x_i)$  当样本$i$，属于第$k$类时，$y_k = 1$， 否则$y_k = 0$

backward:

$\frac{\partial Loss}{\partial x_i} = -x_i^{-1}$ 当$y_k = 1$时，否则梯度为0。

caffe 中的实现如下：

```cpp
template <typename Dtype>
void MultinomialLogisticLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* bottom_label = bottom[1]->cpu_data();
    int num = bottom[0]->num();
    int dim = bottom[0]->count() / bottom[0]->num();
    Dtype loss = 0;
    for (int i = 0; i < num; ++i)
    {
        // 这里先拿到当前真实的label
        int label = static_cast<int>(bottom_label[i]);
        // 然后这里直接去和预测为label（也就是y_k = 1）的类别去做计算，因为其他情况y_k = 0
        Dtype prob = std::max(bottom_data[i * dim + label], Dtype(kLOG_THRESHOLD));
        // 累积loss
        loss -= log(prob);
    }
    // 这里还除了一个N，标准的公式里没有，但是这并不影响梯度的计算
    top[0]->mutable_cpu_data()[0] = loss / num;
}

template <typename Dtype>
void MultinomialLogisticLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
    if (propagate_down[1])
    {
        LOG(FATAL) << this->type() << " Layer cannot backpropagate to label inputs.";
    }
    if (propagate_down[0])
    {
        const Dtype* bottom_data = bottom[0]->cpu_data();
        const Dtype* bottom_label = bottom[1]->cpu_data();
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
        int num = bottom[0]->num();
        int dim = bottom[0]->count() / bottom[0]->num();
        caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
        // 先拿到传播回来的梯度，根据上述推导反向传播的公式 乘-1。（然后再补上正向传播过来的 除以N）
        const Dtype scale = -top[0]->cpu_diff()[0] / num;
        for (int i = 0; i < num; ++i)
        {
            int label = static_cast<int>(bottom_label[i]);
            // 同样这里直接索引到y_k = 1的类别上进行计算
            Dtype prob = std::max(bottom_data[i * dim + label], Dtype(kLOG_THRESHOLD));
            // 这里是公式中的逻辑
            bottom_diff[i * dim + label] = scale / prob;
        }
    }
}
```

### 4.2 softmax_loss_layer

sofxmax loss是由softmax+cross entropy loss组合而成的。

令$x_i$为样本,$N$为样本个数，$K$为类别个数，$p_i$为softmax的输出，其中$p_i = \frac{\exp(x_i)}{\sum_{j = 0}^{K} exp(x_j)}$    。

forward：

$Loss = -\sum_{i =0}^{N} \sum_{k = 0}^{K} y_k*\log(p_i)$

backward:

$\frac{\partial Loss}{\partial x_j} = \frac{\partial Loss}{\partial p_i} * \frac{\partial p_i}{\partial x_j} $

其中：

$\frac{\partial Loss}{\partial p_i} = -\frac{1}{p_i}$

$\frac{\partial p_i}{\partial x_j} = \left\{ \begin{array}{lr} p_i(1 - p_i) &\mathrm{if}(i=j) \\ -p_i^2 &\mathrm{otherwise}  \end{array} \right.$

所以：

$\frac{\partial Loss}{\partial x_j} = \left\{ \begin{array}{lr} p_i - 1 &\mathrm{if}(i=j) \\ p_i &\mathrm{otherwise} \end{array} \right.$     

```cpp
template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    // The forward pass computes the softmax prob values.
    // 首先计算输入的softmax的结果
    // softmax_bottom_vec_里存的就是bottom的引用，softmax_top_vec_是输出的值
    softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
    // prob_里的数据也指向的是softmax_top_vec_里的值
    const Dtype* prob_data = prob_.cpu_data();
    const Dtype* label = bottom[1]->cpu_data();
    int dim = prob_.count() / outer_num_;
    int count = 0;
    Dtype loss = 0;
    for (int i = 0; i < outer_num_; ++i)
    {
        for (int j = 0; j < inner_num_; j++)
        {    
            // 这里拿到真实的label
            const int label_value = static_cast<int>(label[i * inner_num_ + j]);
            if (has_ignore_label_ && label_value == ignore_label_)
            {
                continue;
            }
            DCHECK_GE(label_value, 0);
            DCHECK_LT(label_value, prob_.shape(softmax_axis_));
            // 然后直接索引到y_k = 1的位置，即当前的预测值和真实值对应的类别上进行计算
            loss -= log(std::max(prob_data[i * dim + label_value * inner_num_ + j], Dtype(FLT_MIN)));
            ++count;
        }
    }
    // 这里进行一次规范处理，让loss规则一些
    top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_, count);
    if (top.size() == 2)
    {
        top[1]->ShareData(prob_);
    }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
    if (propagate_down[1])
    {
        LOG(FATAL) << this->type() << " Layer cannot backpropagate to label inputs.";
    }
    if (propagate_down[0])
    {
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
        const Dtype* prob_data = prob_.cpu_data();
        // 这里直接将softmax的输出copy到diff（求导的结果）
        caffe_copy(prob_.count(), prob_data, bottom_diff);
        const Dtype* label = bottom[1]->cpu_data();
        int dim = prob_.count() / outer_num_;
        int count = 0;
        for (int i = 0; i < outer_num_; ++i)
        {
            for (int j = 0; j < inner_num_; ++j)
            {
                const int label_value = static_cast<int>(label[i * inner_num_ + j]);
                // 这些不需要求导
                if (has_ignore_label_ && label_value == ignore_label_)
                {
                    for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c)
                    {
                        bottom_diff[i * dim + c * inner_num_ + j] = 0;
                    }
                }
                else
                {   // 对于y_k = 1的部分需要再额外-1，如公式中所写
                    bottom_diff[i * dim + label_value * inner_num_ + j] -= 1;
                    ++count;
                }
            }
        }
        // Scale gradient
        // 正向传播的地方做的处理
        Dtype loss_weight = top[0]->cpu_diff()[0] / get_normalizer(normalization_, count);
        caffe_scal(prob_.count(), loss_weight, bottom_diff);
    }
}
```

### 4.3 euclidean_loss_layer

欧式空间Loss, $N$ 为样本个数 ， $x^1$ 为预测点，  $x^2$为真值。

forward:

$Loss = \sum_{i=1}^{N} \frac{|x_i^1 - x_i^2|^2}{2N}$

backward:

$\frac{\partial Loss}{\partial x_i^1} = \frac{x_i^1 - x_i^2}{N} $    

$\frac{\partial Loss}{\partial x_i^2} = \frac{-(x_i^1 - x_i^2)}{N}$    

```cpp
template <typename Dtype>
void EuclideanLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    int count = bottom[0]->count();
    // 按照公式中正向传播的过程
    caffe_sub(count, bottom[0]->cpu_data(), bottom[1]->cpu_data(), diff_.mutable_cpu_data());
    Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
    Dtype loss = dot / bottom[0]->num() / Dtype(2);
    top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
    for (int i = 0; i < 2; ++i)
    {
        if (propagate_down[i])
        {    
            // 同样这里也是按照公式进行的计算
            const Dtype sign = (i == 0) ? 1 : -1;
            const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
            caffe_cpu_axpby(bottom[i]->count(), // count
                alpha,                          // alpha
                diff_.cpu_data(),               // a
                Dtype(0),                       // beta
                bottom[i]->mutable_cpu_diff()); // b
        }
    }
}
```

### 4.4. contrastive_loss_layer

一种用于度量学习(Metric Learning)问题的损失函数。它主要用于学习一个可以将相似样本映射到相近的特征空间,而将不相似的样本映射到较远的特征空间的模型。

forward:

$Loss = \frac{1}{2N}\sum_{i - n}^N ((1 - y_i)*D^2_i + y_i*\max(margin - D_i, 0)^2)$ 其中$D$为欧式距离 $D = \|x^1_i - x^2_i\|_2$ ，$y_i$为标签，当两者匹配（相似）的话为0， 否则为1，$margin$为阈值，直观理解当两者不相似的时候距离越远越好。

backward:

$\frac{\partial Loss}{\partial D} = \left\{ \begin{array}{lr} \frac{1}{2N}(1 - y_i)*2D = D/N &\mathrm{if}\;y_i=0 \\ \frac{y_i}{2N}*2max(0, margin - D_i) = (margin - D)/N &\mathrm{else}\; y_i=1 \end{array}\right. $    

$\frac{\partial D}{\partial x^1_i} = x^1_i - x^2_i$

$\frac{\partial D}{\partial x^2_i} = x_i^2 -x^1_i$

```cpp
template <typename Dtype>
void ContrastiveLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    int count = bottom[0]->count();
    // 首先计算两个点之间的距离
    caffe_sub(count,
        bottom[0]->cpu_data(),     // a
        bottom[1]->cpu_data(),     // b
        diff_.mutable_cpu_data()); // a_i-b_i
    const int channels = bottom[0]->channels();
    // 阈值
    Dtype margin = this->layer_param_.contrastive_loss_param().margin();
    bool legacy_version = this->layer_param_.contrastive_loss_param().legacy_version();
    Dtype loss(0.0);
    for (int i = 0; i < bottom[0]->num(); ++i)
    {    // 距离的平方
        dist_sq_.mutable_cpu_data()[i]
            = caffe_cpu_dot(channels, diff_.cpu_data() + (i * channels), diff_.cpu_data() + (i * channels));
        if (static_cast<int>(bottom[2]->cpu_data()[i]))
        { // similar pairs
            // 相似的点对，直接加到loss中来
            loss += dist_sq_.cpu_data()[i];
        }
        else
        { // dissimilar pairs    
            // 不相似的点对，通过阈值来进行划分并计算
            if (legacy_version)
            {
                loss += std::max(margin - dist_sq_.cpu_data()[i], Dtype(0.0));
            }
            else
            {
                Dtype dist = std::max<Dtype>(margin - sqrt(dist_sq_.cpu_data()[i]), Dtype(0.0));
                loss += dist * dist;
            }
        }
    }
    loss = loss / static_cast<Dtype>(bottom[0]->num()) / Dtype(2);
    top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void ContrastiveLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
    Dtype margin = this->layer_param_.contrastive_loss_param().margin();
    bool legacy_version = this->layer_param_.contrastive_loss_param().legacy_version();
    for (int i = 0; i < 2; ++i)
    {
        if (propagate_down[i])
        {
            const Dtype sign = (i == 0) ? 1 : -1;
            // 计算前面的系数
            const Dtype alpha = sign * top[0]->cpu_diff()[0] / static_cast<Dtype>(bottom[i]->num());
            int num = bottom[i]->num();
            int channels = bottom[i]->channels();
            for (int j = 0; j < num; ++j)
            {
                Dtype* bout = bottom[i]->mutable_cpu_diff();
                if (static_cast<int>(bottom[2]->cpu_data()[j]))
                { // similar pairs
                    // 对于相似的点对，直接计算相乘
                    caffe_cpu_axpby(
                        channels, alpha, diff_.cpu_data() + (j * channels), Dtype(0.0), bout + (j * channels));
                }
                else
                { // dissimilar pairs
                    // 不相似的点对，
                    Dtype mdist(0.0);
                    Dtype beta(0.0);
                    if (legacy_version)
                    {
                        mdist = margin - dist_sq_.cpu_data()[j];
                        beta = -alpha;
                    }
                    else
                    {
                        Dtype dist = sqrt(dist_sq_.cpu_data()[j]);
                        mdist = margin - dist;
                        beta = -alpha * mdist / (dist + Dtype(1e-4));
                    }
                    if (mdist > Dtype(0.0))
                    {
                        caffe_cpu_axpby(
                            channels, beta, diff_.cpu_data() + (j * channels), Dtype(0.0), bout + (j * channels));
                    }
                    else
                    {
                        caffe_set(channels, Dtype(0), bout + (j * channels));
                    }
                }
            }
        }
    }
}
```

### 4.5 hinge_loss_layer

hinge loss是多分类任务重的一个loss，同时也是SVM的目标函数。InnerProductLayer + HingeLossLayer => SVM。

其中K为预测的类别个数，N为样本个数，$\delta$ 不同条件下的系数，p为范数(L1, L2)。

$Loss = \sum_{i = 0}^{N}\sum_{k = 0}^{K}\max(0, 1 - \delta_{ln=k}x_{i,k})^p / N$

$\delta = 1 \;or \;-1 $

当预测的值为$x_{i, k}$ ，真实的标签值为$M$，如果$x_{i, k} == M$， 则$\delta=1$ ，否则$\delta = -1$。

```cpp
template <typename Dtype>
void HingeLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* label = bottom[1]->cpu_data();
    int num = bottom[0]->num();
    int count = bottom[0]->count();
    int dim = count / num;
    // 将x_{i,k}拷贝到结果中
    caffe_copy(count, bottom_data, bottom_diff);
    for (int i = 0; i < num; ++i)
    {   // 对满足条件的部分乘-1，x_{i,k} == M
        bottom_diff[i * dim + static_cast<int>(label[i])] *= -1;
    }
    for (int i = 0; i < num; ++i)
    {
        for (int j = 0; j < dim; ++j)
        {   // 先加上公式中的1，然后取max的过程
            bottom_diff[i * dim + j] = std::max(Dtype(0), 1 + bottom_diff[i * dim + j]);
        }
    }
    Dtype* loss = top[0]->mutable_cpu_data();
    // 最后计算一下范数
    switch (this->layer_param_.hinge_loss_param().norm())
    {
    case HingeLossParameter_Norm_L1: loss[0] = caffe_cpu_asum(count, bottom_diff) / num; break;
    case HingeLossParameter_Norm_L2: loss[0] = caffe_cpu_dot(count, bottom_diff, bottom_diff) / num; break;
    default: LOG(FATAL) << "Unknown Norm";
    }
}

template <typename Dtype>
void HingeLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
    if (propagate_down[1])
    {
        LOG(FATAL) << this->type() << " Layer cannot backpropagate to label inputs.";
    }
    if (propagate_down[0])
    {
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
        const Dtype* label = bottom[1]->cpu_data();
        int num = bottom[0]->num();
        int count = bottom[0]->count();
        int dim = count / num;

        for (int i = 0; i < num; ++i)
        {   // 同样对满足条件的部分先乘-1
            bottom_diff[i * dim + static_cast<int>(label[i])] *= -1;
        }
        const Dtype loss_weight = top[0]->cpu_diff()[0];
        switch (this->layer_param_.hinge_loss_param().norm())
        {
        // 对范数进行求导
        case HingeLossParameter_Norm_L1:
            caffe_cpu_sign(count, bottom_diff, bottom_diff);
            caffe_scal(count, loss_weight / num, bottom_diff);
            break;
        case HingeLossParameter_Norm_L2: caffe_scal(count, loss_weight * 2 / num, bottom_diff); break;
        default: LOG(FATAL) << "Unknown Norm";
        }
    }
}
```

### 4.6 infogain_loss_layer

一般用于多分类的损失函数，其中 infogain matrix是用来描述类别之间的关系，对角线上元素都为1，然后相似类别的惩罚小，不相似类别的惩罚大。

forward:

$Loss = -\sum_{i = 0}^{N}\sum_{k = 0}^{K}[InfogainMatrix]_{i, k}*\log(p_{i,k}) / N$

其中，$p_{i,k} = \frac{\exp(x_k)}{\sum_{j = 0}^{K}\exp(x_j)}$ 

backward:

$\frac{\partial p_{i,k}}{\partial x_j} = \left\{ \begin{array}{lr} p_{i,k}(1 - p_{i,k}) &\mathrm{if}(i=j) \\ \ -p_{i,k}^2 &\mathrm{otherwise} \end{array} \right.$

$\frac{\partial Loss}{\partial x_i} = -\sum_{k=0}^K[InfogainMatrix]_k * \frac{\partial p_i}{\partial x_j} / (p_{i, k}*N )$  

```cpp
template <typename Dtype>
void InfogainLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    // The forward pass computes the softmax prob values.    
    // 首先进行对输入进行softmax处理，这里和上面的softmax_loss是一样的内容
    softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
    const Dtype* prob_data = prob_.cpu_data();
    const Dtype* bottom_label = bottom[1]->cpu_data();
    const Dtype* infogain_mat = NULL;
    if (bottom.size() < 3)
    {
        infogain_mat = infogain_.cpu_data();
    }
    else
    {
        infogain_mat = bottom[2]->cpu_data();
    }
    int count = 0;
    Dtype loss = 0;
    for (int i = 0; i < outer_num_; ++i)
    {
        for (int j = 0; j < inner_num_; j++)
        {   // 拿到真实的label
            const int label_value = static_cast<int>(bottom_label[i * inner_num_ + j]);
            if (has_ignore_label_ && label_value == ignore_label_)
            {
                continue;
            }
            DCHECK_GE(label_value, 0);
            DCHECK_LT(label_value, num_labels_);
            for (int l = 0; l < num_labels_; l++)
            {   // 按照上述公式计算loss
                loss -= infogain_mat[label_value * num_labels_ + l]
                    * log(
                        std::max(prob_data[i * inner_num_ * num_labels_ + l * inner_num_ + j], Dtype(kLOG_THRESHOLD)));
            }
            ++count;
        }
    }
    // 进行归一化
    top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_, count);
    if (top.size() == 2)
    {
        top[1]->ShareData(prob_);
    }
}

template <typename Dtype>
void InfogainLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
    if (propagate_down[1])
    {
        LOG(FATAL) << this->type() << " Layer cannot backpropagate to label inputs.";
    }
    if (propagate_down.size() > 2 && propagate_down[2])
    {
        LOG(FATAL) << this->type() << " Layer cannot backpropagate to infogain inputs.";
    }
    if (propagate_down[0])
    {
        const Dtype* prob_data = prob_.cpu_data();
        const Dtype* bottom_label = bottom[1]->cpu_data();
        const Dtype* infogain_mat = NULL;
        if (bottom.size() < 3)
        {
            infogain_mat = infogain_.cpu_data();
        }
        else
        {
            infogain_mat = bottom[2]->cpu_data();
            // H is provided as a "bottom" and might change. sum rows every time.
            sum_rows_of_H(bottom[2]);
        }
        const Dtype* sum_rows_H = sum_rows_H_.cpu_data();
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
        const int dim = bottom[0]->count() / outer_num_;
        int count = 0;
        for (int i = 0; i < outer_num_; ++i)
        {
            for (int j = 0; j < inner_num_; ++j)
            {
                const int label_value = static_cast<int>(bottom_label[i * inner_num_ + j]);
                DCHECK_GE(label_value, 0);
                DCHECK_LT(label_value, num_labels_);
                if (has_ignore_label_ && label_value == ignore_label_)
                {
                    for (int l = 0; l < num_labels_; ++l)
                    {
                        bottom_diff[i * dim + l * inner_num_ + j] = 0;
                    }
                }
                else
                {
                    for (int l = 0; l < num_labels_; ++l)
                    {   // 这里实际上是对i==j 和 i!=j的两个情况进行了合并
                        // i==j：infogain_mat * (p - 1)
                        // i!=j: infogain_mat * p
                        // 合并之后就是 sum_rows_H * p - informat[i==j]
                        bottom_diff[i * dim + l * inner_num_ + j]
                            = prob_data[i * dim + l * inner_num_ + j] * sum_rows_H[label_value]
                            - infogain_mat[label_value * num_labels_ + l];
                    }
                    ++count;
                }
            }
        }
        // Scale gradient
        Dtype loss_weight = top[0]->cpu_diff()[0] / get_normalizer(normalization_, count);
        caffe_scal(bottom[0]->count(), loss_weight, bottom_diff);
    }
}
```

### 4.4. sigmoid_cross_entropy_loss_layer

与softmax loss不同的是，sigmoid loss通常适用于预测概率的输出。

forward:(其中$\hat{p_i}$为sigmoid的输出，$p_i$为真实值， $\hat{p_i} = sigmoid(x_i) = \frac{1}{1+\exp(-x_i)}$)

$Loss = -\frac{1}{N}\sum_{i=0}^{N} [p_i*\log(\hat{p_i}) + (1 - p_i)*\log(1 - \hat{p_i})]$    

实际在计算的过程中为了满足计算的健壮性对公式进行了变化：

[Caffe Loss层 - SigmoidCrossEntropyLossLayer_caffe sigmoidcrossentroylosslayer-CSDN博客](https://blog.csdn.net/zziahgf/article/details/77161648)

<img title="" src="file:///home/aklice/.config/marktext/images/2024-06-19-16-42-51-layers_sigmoid_cross_entropy.png" alt="" width="465" data-align="center">

backward:

$\frac{\partial Loss}{\partial x_i} = \frac{1}{N}(\hat{p_i} - p_i)$    

```cpp
template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
    // The forward pass computes the sigmoid outputs.
    sigmoid_bottom_vec_[0] = bottom[0];
    sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
    // Compute the loss (negative log likelihood)
    // Stable version of loss computation from input data
    const Dtype* input_data = bottom[0]->cpu_data();
    const Dtype* target = bottom[1]->cpu_data();
    int valid_count = 0;
    Dtype loss = 0;
    for (int i = 0; i < bottom[0]->count(); ++i)
    {
        const int target_value = static_cast<int>(target[i]);
        if (has_ignore_label_ && target_value == ignore_label_)
        {
            continue;
        }
        loss -= input_data[i] * (target[i] - (input_data[i] >= 0))
            - log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0)));
        ++valid_count;
    }
    normalizer_ = get_normalizer(normalization_, valid_count);
    top[0]->mutable_cpu_data()[0] = loss / normalizer_;
}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
    if (propagate_down[1])
    {
        LOG(FATAL) << this->type() << " Layer cannot backpropagate to label inputs.";
    }
    if (propagate_down[0])
    {
        // First, compute the diff
        const int count = bottom[0]->count();
        const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
        const Dtype* target = bottom[1]->cpu_data();
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
        caffe_sub(count, sigmoid_output_data, target, bottom_diff);
        // Zero out gradient of ignored targets.
        if (has_ignore_label_)
        {
            for (int i = 0; i < count; ++i)
            {
                const int target_value = static_cast<int>(target[i]);
                if (target_value == ignore_label_)
                {
                    bottom_diff[i] = 0;
                }
            }
        }
        // Scale down gradient
        Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer_;
        caffe_scal(count, loss_weight, bottom_diff);
    }
}
```

## 5. Recurrent Layers

[The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) 

在了解循环神经网络之前，看一下先简单看一下关于RNN的介绍。

循环神经网络是一种在序列上的网络，引入了隐状态这种神经节元，对于一个节点来说，它的输入包括当前的数据和前一个节点的信息输入，使得每个神经元的参数共享。caffe中实现RNN的时候就是将这样一个神经元抽象成为一个layer，上一层的输出，会作为当前layer的输出。

### 5.1 RecurrentLayer

RecurrentLayer就是将RNN这种结构进行抽象，其内部unroll_net包含了RNN的网络结构，然后将隐状态之前的输入输出进行联系。

### 5.2  RNNLayer

### 5.3 LSTMLayer

## 6. Other Layers

### 6.1 BatchNormalize

### 6.2 LayerNormalize

### 6.3 Group Normalize
