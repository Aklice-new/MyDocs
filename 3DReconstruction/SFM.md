# Structure from Motion

## Problem definition and ambiguities

<img title="" src="file:///home/aklice/.config/marktext/images/2024-05-30-20-35-50-sfm_problem.png" alt="" width="384" data-align="center">

给定几个已知的相机，然后每张图像中有一些共视点，如何估计出来相机的**内外参数**。

## Affine structure from motion



## Projective structure from motion



## Modern structure from motion pipeline



- Feature Detection （SIFT， SURF， ORB， BRISK）

- Feature Matching    RANSAN)

- Image connectivity graph

- Incremental SFM 
  
  - 选择一对有很多匹配内点图像
    
    - 初始化内参
    
    - 通过5点法估计外参
    
    - 通过三角化来初始化模型中的一些点
  
  - 循环所有图像
    
    - 选择一张与建立的模型有尽可能多的匹配点的图像
    
    - 通过RANSAC将这些匹配的特征和模型进行配准
    
    - 三角化新的点
    
    - BA优化相机参数和位姿
    
    - 额外的，可以通过一些其他辅助的数据如GPS数据来进行对齐

- 注意：在SFM过程中，如果碰到一些重复的结构可能会导致很严重的问题。

- Reducing error accumulation and closing loops 通过回环检测来减少误差。


