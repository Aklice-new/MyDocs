

# 第1章

## CUDA调试方法
1. CUDA-gdb ：在nvcc 的flag中指定-g 表示主机代码编译为可调试版本， -G表示GPU代码编译为可调式版本。《CDUA-GDB NVIDIA CUDA Debugger for Linux and Mac》中提供了一些细节信息。 《Doctor Dobb's Journal》。
2. cuda-memcheck 可以正确标出存在一处内存越界访问错误。