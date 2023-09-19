
## 描述

这个示例说明了CUDA事件在GPU计时和重叠CPU和GPU执行中的使用。事件被插入到CUDA调用流中。由于CUDA流调用是异步的，CPU可以在GPU执行时执行计算(包括主机和设备之间的DMA memcopy)。CPU可以查询CUDA事件，判断GPU是否完成任务。

## 关键概念

异步数据传输，CUDA流和事件


## 代码细节


## 执行结果
![[asyncAPI.png]]