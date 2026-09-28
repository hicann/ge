# 简介

`gert::ResourceUsageContext`继承自`gert::ExtendedKernelContext`，是[`ge::ResourceUsageReporter::DeclareResourceUsage`](../../ge/ResourceUsageReporter/DeclareResourceUsage.md)的资源申报上下文。

## 需要包含的头文件

```c++
#include <exe_graph/runtime/resource_usage_context.h>
```

库文件：liblowering.so

## Public成员函数

| 函数 | 功能 |
| --- | --- |
| [`ge::graphStatus ReportAttachedStream(const std::vector<ge::AscendString> &keys)`](ReportAttachedStream.md) | 上报本节点运行期将请求的辅流key列表。 |
| [`const Tensor *GetInputTensor(size_t index) const`](GetInputTensor.md) | 按扁平实例索引获取输入Tensor。 |
| [`const Tensor *GetOutputTensor(size_t index) const`](GetOutputTensor.md) | 按扁平实例索引获取输出Tensor。 |
| [`const Tensor *GetRequiredInputTensor(size_t ir_index) const`](GetRequiredInputTensor.md) | 按IR原型索引获取必选输入Tensor。 |
| [`const Tensor *GetOptionalInputTensor(size_t ir_index) const`](GetOptionalInputTensor.md) | 按IR原型索引获取可选输入Tensor。 |
| [`const Tensor *GetDynamicInputTensor(size_t ir_index, size_t relative_index) const`](GetDynamicInputTensor.md) | 按IR原型索引和相对索引获取动态输入Tensor。 |
| [`const Tensor *GetRequiredOutputTensor(size_t ir_index) const`](GetRequiredOutputTensor.md) | 按IR原型索引获取必选输出Tensor。 |
| [`const Tensor *GetDynamicOutputTensor(size_t ir_index, size_t relative_index) const`](GetDynamicOutputTensor.md) | 按IR原型索引和相对索引获取动态输出Tensor。 |

## 相关接口

- [`ge::ResourceUsageReporter`](../../ge/ResourceUsageReporter/overview.md)
- [`gert::EagerOpExecutionContext::RequestAttachedStream`](../EagerOpExecutionContext/RequestAttachedStream.md)
- [`CompiledGraphSummary::GetStreamNum`](../../ge/CompiledGraphSummary/GetStreamNum.md)
