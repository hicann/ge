# 简介

`ge::ResourceUsageReporter`是自定义算子的资源申报接口，算子实现`DeclareResourceUsage`后，GE会在图编译阶段按节点回调该接口，由算子申报本节点运行期将使用资源。

该接口面向Eager类自定义算子（实现[`EagerExecuteOp`](../EagerExecuteOp/overview.md)并在`Execute`中申请辅流）的辅流资源观测场景：

- 统计是纯编译期旁路观测能力，不改变流分配、内存规划与运行期执行行为；统计值不是资源配额或预留承诺。
- 未实现该接口的算子不参与统计，编译与执行不受影响。
- 声明式算子（`ArgsRefreshStrategy`为`kAnnotatedArgs`的节点）的资源使用由框架精确计数，GE不会对此类节点回调`DeclareResourceUsage`；即使算子同时实现`AnnotatedArgsOp`与`ResourceUsageReporter`，也按声明式路径处理，避免重复计入。

## 需要包含的头文件

```c++
#include <graph/custom_op.h>
```

## Public成员函数

```c++
virtual graphStatus DeclareResourceUsage(gert::ResourceUsageContext &ctx) = 0
```

## 相关接口

- [`DeclareResourceUsage`](DeclareResourceUsage.md)
- [`gert::ResourceUsageContext`](../../gert/ResourceUsageContext/overview.md)
- [`gert::EagerOpExecutionContext::RequestAttachedStream`](../../gert/EagerOpExecutionContext/RequestAttachedStream.md)
- [`CompiledGraphSummary::GetStreamNum`](../CompiledGraphSummary/GetStreamNum.md)
