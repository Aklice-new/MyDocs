# TinyRender 记录

 tinyrender是对opengl的一个最简单的实现，其目的就是剖析一下opengl中各个具体的流程。现代图形渲染管线包括以下几个步骤：(模型加载->) 顶点着色器 -> 图元装配 -> 光栅化 -> 片段着色器 -> 深度混合 -> 输出图像。该篇记录在tinyrender的基础上，记录一下部分步骤中的数学推导过程。

## 模型和纹理加载

模型是以obj保存的，obj文件通常与mtl文件一起进行使用，mtl文件包含了模型表面贴图的信息包括材质、反射系数等。但是该项目中使用tga这种位图的形式来保存这些信息。

obj文件存有一下内容：

1. ‘v’开头的行表示顶点的三维坐标， 由三个浮点数表示。

2. ‘vt’开头的行表示纹理的uv坐标， 由两个浮点数表示，vt 的索引顺序和v的索引顺序严格一致。

3. 'vn'开头的行表示顶点的法向量，由三个数表示，用于计算光照相关的信息。

4. ‘f’开头的行表一个三角形面片，一个三角形由三个点构成，所以会以三个顶点索引的方式给出。例：f 3/3/1 2/2/1 1/1/1.即为三个顶点的索引。

纹理一般是和模型分开保存的，模型只需要保存与纹理文件的映射关系，如obj中的‘vt’属性。这里使用的是tga格式的文件。

## 顶点着色器

顶点着色器就是对模型的顶点进行线性变换。

1. 从模型中加载完顶点之后，这个时候顶点坐标是在模型的局部坐标系，一般来说，首先要通过一次线性变换，将模型变换到当前的世界坐标系下（这里也可以认为是对模型进行缩放 平移和旋转到世界坐标系下的某个位置），这里的世界坐标系是opengl中的坐标系。

2. 接下来我们我们要渲染图像，所以需要空间中一个相机的位置，以及这个相机的朝向(一般我们用外参来描述相机在空间的中的位置和朝向)，为了将世界空间中的点转换到相机的坐标系下，这里还需要一个变换矩阵，我们叫做lookat矩阵，构造这个矩阵需要给出三个信息：相机在世界坐标系下的位置(camera_position)，相机的目标(target_position)位置以及相机上方向(up_vector)。
   
   下面是具体的推导过程：
   
   先定义一下已知的量，$P_c$ 表示相机在世界坐标系下的位置，$P_t$表示相机的目标位置，$up$表示相机上方向。我们的目的是将世界坐标系下的点变换到相机坐标系下。这个过程包含了旋转和平移两个过程。
   
   首先来看旋转的这个过程，对于空间中一点$P [a_1, a_2, a_3]$，它在世界坐标系下一组基$[e_1, e_2, e_3]$下可以由线性组合表示，同样它可以在相机坐标系下用另一组基$[e_1', e_2', e_3']$来表示，即：
   
   $$
   [e_1, e_2, e_3] * [a_1, a_2, a_3]^{T}  = [e_1', e_2', e_3'] * [a_1', a_2', a_3']^{T}
   $$
   
   其中因为$e$这组基表示的是世界坐标系，所以可以直接写出来这组基：
   
   $$
   e_1 = [1, 0, 0] \\
e_2 = [0, 1, 0] \\
e_3 = [0, 0, 1]
   $$

 为了求得$[a_1', a_2', a_3']^{T}$，根据矩阵论相关知识，我们需要求得两个基之间的变换矩阵$M$。

那么对于相机空间，我们可以根据$P_c, P_t, up$这三个向量来求得相机空间的一组基(满足正交性)。

首先根据相机的位置$P_c$和$P_t$我们可以确定相机空间的z轴方向，即为$z = \overrightarrow{P_tP_c}$， 然后根据相机的$up$上方向向量可以叉乘出来y轴的方向$y = \overrightarrow{P_tP_c}\times up$ ， 然后可以根据z轴和y轴叉乘出来x轴的方向$x = z \times y$。所以可以写出下面旋转变换的式子：

$$
M = R_{c2w} = [x^T, y^T, z^T]
$$

    然后还有平移变换，对于平移变换就比较简单，经过旋转变换后，世界坐标系和相机坐标系的原点是重合的，现在将相机坐标系平移到对应位置。所以平移变换的式子为：

$$
T_{w2c} = P_c^T
$$

    所以整个公式x写成齐次坐标就是：

$$
lookat = (R_{w2c}\times T_{w2c}) = \begin{bmatrix}
  x_1& x_2 & x_3 & 0\\
  y_1& y_2 & y_3 & 0 \\
  z_1& z_2 & z_3 & 0 \\
  0&  0&  0&1
\end{bmatrix} * \begin{bmatrix}
 1 &  0&  0& -P_{c,x}\\
 0 &  1&  0& -P_{c,y}\\
 0 &  0&  1& -P_{c,z}\\
 0 &  0&  0& 1
\end{bmatrix}
$$

3. 接着会对模型进行透视变换(Perspective transforms)，

    透视变换就是根据“近大远小”的成像规律对相机坐标系中的点进行变换，将三维空间中的点$x, y, z$变换为成像平面上的一点$u, v$,透变换不是线性变换。透视变换矩阵通常是以4x4的矩阵给出。

$$
[u, v, 1]^T = \begin{bmatrix}
 1& 0& 0 & 0\\
 0& 1& 0& 0\\
 0& 1& 0& 0\\
 0& 0& \frac{1}{f}&0
\end{bmatrix}  * [x, y, z]^T
$$

4. 然后对点进行裁剪，为了将透视投影之后平面上的点(这里已经转为NDC规范化设备坐标系)转换为屏幕空间上的点。

    NDC坐标系中的元素满足$\left\{\begin{aligned} -1 \le x \le 1 \\ -1 \le y \le 1 \\ 0 \le z \le 1\end{aligned}\right.$    

转换为屏幕空间（宽为w，高为h，视口的起点为(X,Y)上对应点的坐标为$\left\{ \begin{aligned}X \le x \le X+Width \\ Y \le y \le Y + Height \\ MinZ\le z \le MaxZ\end{aligned}\right. $        

整个变换过程可以变为以下几个步骤：

1. 平移，将点从$[-1, 1]$平移到$[0, 2]$的范围来。

2. 缩放，从$[0, 2]$缩放到$[0, 1]$。

3. 缩放，从$[0,1]$缩放到$[0,w], [0, h], [0,d]$。

4. 平移，从$[0,w],[0,h],[0,d]$ 平移到$[X, X + w],[Y, Y + h],[0,d]$。

合并起来就是下面的矩阵：

$$
\begin{bmatrix}
  \frac{w}{2}&  0&  0& x + \frac{w}{2}\\
  0&  \frac{h}{2}&  0& y+\frac{h}{2}\\
  0&  0&  \frac{d}{2}& \frac{d}{2}\\
  0&  0&  0& 1
\end{bmatrix}
$$

以上就是顶点着色器的所有过程，完成了对模型的每一个顶点(3D)转换到最后成像的图片的像素(2D)的过程。

## 图元装配

图元装配是在顶点着色器完成之后对已经变换后的顶点进行装配，例如，线就是将两个点进行装配等等。装配方式决定了最后的显示效果：

<img title="" src="file:./img/Assembly.png" alt="" width="374" data-align="center">

## 光栅化

光栅化就是计算投影在屏幕上的这些点，构成的三角形(我们以三角形为例)具体覆盖了哪些像素。

一般的计算过程为：

1. 计算包含该三角形的最小矩形(bounding box)。

2. 判断这个像素点是否在这个三角形内。这里具体判断的时候使用的是重心坐标去判断。

3. 通过重心坐标去做的插值，然后将该像素对应到3D空间中的坐标计算出来。

给定一个三角形 ABC 和一个点 P，我们希望求出点 P 相对于三角形的重心坐标 (u, v, w)。重心坐标满足以下性质:

1. u + v + w = 1，表示三个坐标的和为 1。
2. 如果 P 在三角形内部，则 u, v, w 都在 [0, 1] 区间内。
3. 如果 P 在三角形的边上，那么有一个坐标为 0。
4. 如果 P 在三角形的顶点上，那么有两个坐标为 0。

所以首先会将三角形的三个顶点，从3D投影到2D平面，然后再去计算重心坐标。

之后我们再将这个2D的中心坐标重投影回3D空间中，用于计算该像素插值之后的信息。

设$A, B,C,P$分别是三角形的三个顶点和三角形中的一个点，其投影之后的点为$A',B',C',P'$，通过透视投影变换建立两者的联系:
$$
P' = \frac{1}{z*r + 1}*P
$$
其中z表示点在三维空间中的深度，r是一个常数因子，用于调整透视的程度。

设在3D空间中的重心坐标为$\alpha, \beta, \gamma$， 投影之后2D空间中的重心坐标为$\alpha',\beta',\gamma'$。则有：

$$
P = [A,B,C][\alpha, \beta,\gamma]^T \\ P' = [A',B',C'][\alpha', \beta',\gamma']^T\\
$$

联合得:

$$
P*\frac{1}{P_z*r +1} = 
[A*\frac{1}{A_z*r+1}, B*\frac{1}{B_z*r+1},C*\frac{1}{C_z*r+1}] * [\alpha', \beta',\gamma']^T
$$

$$
[\alpha, \beta,\gamma]^T 
= [\frac{1}{A_z*r +1}, \frac{1}{B_z*r +1},\frac{1}{C_z*r +1}] * (P_z*r+1)*[\alpha',\beta',\gamma']^T
$$

同时还得满足:$\alpha + \beta + \gamma =1$

所以

$$
[\frac{1}{A_z*r +1}, \frac{1}{B_z*r +1},\frac{1}{C_z*r +1}] * (P_z*r+1) = 1
$$

因为我们实际上是不知道$P_z$的值，但是它只是一个常数，最后要满足$\alpha +\beta+\gamma = 1$这个条件，可以最后做一次归一化处理。

这样我们就得到了每个像素在3D中的重心坐标，就可以丢给片段着色器去给像素处理，根据重心坐标插值颜色法线等属性，然后给像素上色。

在实际的处理过程中，由于三角形边缘的一些像素被部分覆盖，所以一些情况可能会选择保留这些像素或者删除，但是这样的结果会有很强的锯齿效果。出现这种不正常的现象，我们称之为走样，常见的表现有锯齿、摩尔纹等都是走样的表现。其本质是因为"信号的频率太快，而采样的频率太慢"。光栅化就是对屏幕中二维的图形的采样。针对这个问题，可以先对图像做一次模糊处理，然后再进行采样，这是从信号处理的角度进行解决的。在相同的采样率的情况下，对高频信号的采样结果并没有对低频信号的采样结果准确。所以我们会进行一次模糊处理，将高频信号处理为低频信号。锯齿的出现是由于采样率不足，为了提高采样率，可以通过提升分辨率、反走样(计算实际每个像素的覆盖率，但实际比较难处理)和超采样(每个像素点内有多个采样点，用于计算对这个像素的覆盖率)。

<img title="" src="file:./img/alias.png" alt="" data-align="center" width="270">

## 片段着色器

### 颜色计算

通过计算得到的重心坐标，对该像素点的uv坐标和法相量进行插值计算。

首先计算uv坐标的插值，在模型加载过程中obj文件中提供了顶点和贴图文件中坐标的对应关系，于是我们可以得到三个顶点的uv坐标和重心坐标$\alpha, \beta, \gamma$然后通过:

$$
uv_{interpolated} = \alpha * uv_1 + \beta * uv_2 + \gamma * uv_3
$$

计算可以得到插值后的结果。

### 光照计算

光照和像素点的法向量有关系。我们在加载模型的时候获取到各个顶点的法向量，这个法向量是在模型坐标系下的。在渲染过程中，同一个平面的法线只有一个垂直于该平面的法向量，所以该平面被以一致的方向照亮。这样的光照很假，我们希望的是足够真实的，所以需要每个像素的法向量是不相同的，这样渲染的光照更真实。

<img title="" src="file:./img/fragment_shader.png" alt="" width="613" data-align="center">

为了使得各个像素点的法相量不同，我们完全可以用一个texture来保存每个点的法向量(r,g,b三个元素来代表法向量的三个值)。但是此时这个法向量存在一个问题，它是固定反向的，当模型发生变换，图元的表面法相也会发生变换，但是法线贴图并不会变化，这样就会发生问题。

#### 切线空间

为了解决这个问题，我们定义了切线空间。在这个坐标空间中，法线贴图向量永远指向这个空间的z轴方向，然后再将这个法线转换到世界坐标系下，使它转向到最终贴图的表面方向。

转换坐标系的这个矩阵叫做TBN矩阵(tangent,bitangent,normal)，用这三个向量来构造这个转换矩阵，这三个向量是相互垂直的。其具体的过程和之前求lookat矩阵的数学原理一致，都是线性空间的变换。<img title="" src="file:./img/tangent_space_1.png" alt="" width="244" data-align="center">

首先如图所示的是切线空间，其中T为右方向，B为前方向，N为上方向。所以我们可以构造出来切线空间的这组基$[T,B,N]^T$。

其中规定了B和T一定是沿着uv的坐标轴的，所以我们可以根据这一性质来求得B和T向量。

<img title="" src="file:./img/tangent_space_2.png" alt="" width="266" data-align="center">

 其中$E_1,E_2$可以通过$T,B$的线性组合来表示，$T,B$是这个平面空间的基，其中$\Delta U,\Delta V$是纹理点的差。

$$
E_1 = \Delta U_1T + \Delta V_1 B \\
E_2 = \Delta U_2T + \Delta V_2 B
$$

同样可以写成矩阵的形式：

$$
\begin{bmatrix} 
 E_{1x} & E_{1y} & E_{1_z} \\
 E_{2x} & E_{2y} & E_{2_z}
\end{bmatrix}  
= \begin{bmatrix} 
 \Delta U_1 & \Delta V_1 \\
 \Delta U_2 & \Delta V_2
\end{bmatrix} 
* 
 \begin{bmatrix} 
 T_x & T_y & T_z \\
 B_x & B_y & B_z
\end{bmatrix}  
$$

可以求得$[T,B]^T$的结果，通过线性方程组的方法，$AX = B, X = A^{-1}B$

将法线贴图中的法向量从切线空间转换到世界空间中（相机空间）。

然后再带上环境光、漫反射和镜面反射的光照(这些都是通过光源光线方向、法线和人眼的入射光线来决定的)。

这样就完成了一个像素的着色。

## 最后

完整的渲染管线当然比这个复杂的多，本篇文章只是基于tinyrender这个项目记录一下其中一些数学公式推导的过程。
