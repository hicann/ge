# 简介

`ge::AnnotatedArgsOp`是自定义算子的声明式kernel launch参数接口，虚继承自`ge::BaseCustomOp`。算子实现`DeclareLaunchArgs`后，GE会在静态图编译阶段回调该接口，由算子通过[`gert::AnnotatedArgsContext`](../../gert/AnnotatedArgsContext/overview.md)获取Tensor、申请workspace，并提交kernel launch。

<!-- npu="x90,9030" id1 -->
端侧场景：只允许提交一个kernel launch。
<!-- end id1 -->

该接口用于静态图任务生成。动态图不走声明式参数生成路径；已生成的`args_format`会在运行时用于地址刷新，运行时不会再次回调`DeclareLaunchArgs`。

如果同一个自定义算子同时继承[`ArgsUpdater`](../ArgsUpdater/overview.md)和`AnnotatedArgsOp`，GE选择`ArgsUpdater`的回调刷新策略，`DeclareLaunchArgs`不参与地址刷新。

## 需要包含的头文件

```c++
#include <graph/custom_op.h>
```

## Public成员函数

```c++
virtual graphStatus DeclareLaunchArgs(gert::AnnotatedArgsContext &ctx) = 0
```

## 相关接口

- [`DeclareLaunchArgs`](DeclareLaunchArgs.md)
- [`gert::AnnotatedArgsContext`](../../gert/AnnotatedArgsContext/overview.md)
- [`gert::AnnotatedKernelArgs`](../../gert/AnnotatedKernelArgs/overview.md)
