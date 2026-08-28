# 简介

自定义算子的基类，用于在host实现自定义的操作，通常可在常量折叠或运行时Host CPU调度场景完成计算。

## 需要包含的头文件

```c++
#include <graph/custom_op.h>
```

## Public成员函数

```c++
virtual graphStatus Execute(gert::HostCpuOpExecutionContext *ctx) = 0
```
