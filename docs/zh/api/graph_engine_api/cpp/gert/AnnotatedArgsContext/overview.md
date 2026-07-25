# 简介

`gert::AnnotatedArgsContext`继承自`gert::ExtendedKernelContext`，是[`ge::AnnotatedArgsOp::DeclareLaunchArgs`](../../ge/AnnotatedArgsOp/DeclareLaunchArgs.md)的声明式参数上下文。

Context及其返回的Tensor指针只在当前`DeclareLaunchArgs`回调期间有效。调用方不得将Context、Tensor指针或其中的逻辑地址缓存到`DeclareLaunchArgs`回调之外。

从Context接口中获取到的输入输出地址，stream ID以及通过`MallocWorkSpace`申请的workspace内存地址均为逻辑资源，而非实际物理资源。

## 需要包含的头文件

```c++
#include <exe_graph/runtime/annotated_args_context.h>
```

库文件：liblowering.so

## Public成员函数

| 函数 | 功能 |
| --- | --- |
| [`WorkspaceAddr MallocWorkSpace(size_t size)`](MallocWorkSpace.md) | 申请逻辑workspace。 |
| [`uint32_t GetStreamId() const`](GetStreamId.md) | 获取节点的逻辑主stream ID。 |
| [`ge::graphStatus AddLaunch(const AnnotatedKernelLaunchInfo &launch_info, AnnotatedKernelArgs &&args)`](AddLaunch.md) | 添加一个kernel launch。 |
| [`const Tensor *GetInputTensor(size_t index) const`](GetInputTensor.md) | 按扁平实例索引获取输入Tensor。 |
| [`const Tensor *GetOutputTensor(size_t index) const`](GetOutputTensor.md) | 按扁平实例索引获取输出Tensor。 |
| [`const Tensor *GetRequiredInputTensor(size_t ir_index) const`](GetRequiredInputTensor.md) | 按IR原型索引获取必选输入Tensor。 |
| [`const Tensor *GetOptionalInputTensor(size_t ir_index) const`](GetOptionalInputTensor.md) | 按IR原型索引获取可选输入Tensor。 |
| [`const Tensor *GetDynamicInputTensor(size_t ir_index, size_t relative_index) const`](GetDynamicInputTensor.md) | 按IR原型索引和相对索引获取动态输入Tensor。 |
| [`const Tensor *GetRequiredOutputTensor(size_t ir_index) const`](GetRequiredOutputTensor.md) | 按IR原型索引获取必选输出Tensor。 |
| [`const Tensor *GetDynamicOutputTensor(size_t ir_index, size_t relative_index) const`](GetDynamicOutputTensor.md) | 按IR原型索引和相对索引获取动态输出Tensor。 |

## 相关类型

- [`gert::AnnotatedKernelArgs`](../AnnotatedKernelArgs/overview.md)
- [`gert::InputAddr`](../InputAddr.md)
- [`gert::OutputAddr`](../OutputAddr.md)
- [`gert::WorkspaceAddr`](../WorkspaceAddr.md)
- [`gert::AnnotatedKernelLaunchInfo`](../AnnotatedKernelLaunchInfo.md)
