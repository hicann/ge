# GetOutputTensor

## 产品支持情况

全量芯片支持。

## 头文件/库文件

- 头文件：\#include <exe\_graph/runtime/host\_cpu\_op\_execution\_context.h\>
- 库文件：liblowering.so

## 功能说明

获取index指定的输出Tensor指针。

## 函数原型

```c++
const Tensor *GetOutputTensor(size_t index) const
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| index | 输入 | 输出索引。 |

## 返回值说明

输出Tensor指针，异常时返回空指针。

## 约束说明

无

## 调用示例

以下片段位于`HostCpuExecuteOp::Execute`实现中，获取第0个输出Tensor，并在使用前检查返回值。

```c++
ge::graphStatus Execute(gert::HostCpuOpExecutionContext *ctx) override {
  const gert::Tensor *output = ctx->GetOutputTensor(0U);
  if (output == nullptr) {
    return ge::GRAPH_FAILED;
  }

  // 使用output获取输出Tensor的描述信息或数据。
  return ge::GRAPH_SUCCESS;
}
```
