## 球谐函数

傅里叶变换：$$f(x) = a_0 + \sum_{n = 1}^{+\inf}a_ncos(\frac{n\pi}{l})x + b_nsin\frac{n\pi}{l}x$$
任意函数都可以用这个函数来进行完美拟合
我们考虑随意一个封闭、有界、光滑的曲面：
![[3d_gaussian_1.png]]
它也可以用球面函数方法来表示$f(\theta, \phi)$，同样也能用基函数来表示，因此，任意一个球面坐标的函数就可以用多个球谐函数来近似。左侧是level。 
![[3d_gaussian_2.png]]
## 3D Gaussian
99%落在$\mu - 3\sigma$ 内，三维的同样也是。



## cuda 代码学习

## diff_gaussian_rasterizatoin/cuda_rasterzier/forward.cu 前向传播部分
 整个前向传播的过程为：
1. 计算Gaussian球投影出来的近似圆
2. 计算这个圆覆盖了哪些像素格子
3. 计算每个gaussian球覆盖像素格子的前后顺序(GPU基数排序)
4. 计算每个像素的颜色  forward.cu renderCUDA()
其中preprocessCUDA函数包含了前两个部分的内容。

最后得到的是render之后的彩色图像。
## diff_gaussian_rasterizatoin/cuda_rasterzier/forward.cu 反向传播部分
- output:  
p_rgb(P, 3)  p = pixel
- inputs:  
g_point 3dposition (G, 3)
g_rgb color  (G, 3)
g_rotation 四元数 (G, 4)
g_scale 三个方向轴的长度 (G, 3)
g_opacity 透明度 (G, 1)

Loss 是定义的损失项， L1是gr和render出来的图片的误差。
反向传播的值就是$\frac{dLoss}{dP_{rgb}}$， $P_{rgb}$是$P * 3$的tensor，
所以要求偏微分：$\frac{dLoss}{dG_{position}},\frac{dLoss}{dG_{position}}, \frac{dLoss}{dG_{rgb}},\frac{dLoss}{dG_{rotation}},\frac{dLoss}{dG_{scale}},\frac{dLoss}{dG_{opacity}}$
通过链式求导法则，



