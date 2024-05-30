# MVS: MultiView Stereo 多视图立体视觉

以下内容多来自于该篇[文章](https://slazebni.cs.illinois.edu/fall22/lec20_multiview_stereo.pdf)。一般是在SFM的基础上，得到相机的位姿。

## 平面扫描算法 Plane Sweeping

<img src="file:///home/aklice/.config/marktext/images/2024-05-30-13-29-32-plane_sweeping.png" title="" alt="" data-align="inline">

给定多个相机，然后通过以主相机为主，依次的多个平面（不同深度）进行测试，来得到每个像素点的深度值。

<img src="file:///home/aklice/.config/marktext/images/2024-05-30-13-35-50-projection_1.png" title="" alt="" width="247">         <img src="file:///home/aklice/.config/marktext/images/2024-05-30-13-36-06-projection_2.png" title="" alt="" width="252">         <img src="file:///home/aklice/.config/marktext/images/2024-05-30-13-36-13-projection_3.png" title="" alt="" width="250">

每一个深度值都对应一个平面，将所有的图像通过Homography变换到该平面，然后计算对应的像素值的方差，认为方差最小的深度值即为正确的深度值。

<img title="" src="file:///home/aklice/.config/marktext/images/2024-05-30-13-36-58-projection_4.png" alt="" width="364" data-align="center">

选择其中方差最小的一个深度值，作为该点的深度值。

具体的公式推导可以参考博客([平面扫描 | Plane Sweeping - 技术刘](http://liuxiao.org/kb/3dvision/3d-reconstruction/%E5%B9%B3%E9%9D%A2%E6%89%AB%E6%8F%8F-plane-sweeping/))

## 深度值融合 Depth Fusion

来生成Mesh或者Voxel。

通过融合多个视角的depth图，来得到sdf，然后再得到通过marching cube得到mesh。



## Patch-based multi-view stereo(PMVS)

1. Detect keypoints
2. Triangulate a sparse set of initial matches
3. Iteratively expand matches to nearby locations
4. Use visibility constraints to filter out false matches
5. Perform surface reconstruction







## DeepLearning For MVS

<img src="file:///home/aklice/.config/marktext/images/2024-05-30-16-35-37-dl_with_sfm_mvs_1.png" title="" alt="" width="425">                <img src="file:///home/aklice/.config/marktext/images/2024-05-30-16-35-44-dl_with_sfm_mvs_2.png" title="" alt="" width="371"> 

## DeepLearning For SFM

![](/home/aklice/.config/marktext/images/2024-05-30-16-36-06-dl_with_sfm_mvs_3.png)


