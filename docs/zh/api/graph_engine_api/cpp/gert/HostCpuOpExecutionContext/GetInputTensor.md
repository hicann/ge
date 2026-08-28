# GetInputTensor

## 产品支持情况

全量芯片支持。

## 头文件/库文件

- 头文件：\#include <exe\_graph/runtime/host\_cpu\_op\_execution\_context.h\>
- 库文件：liblowering.so

## 功能说明

根据输入index，获取输入tensor指针。

## 函数原型

```c++
const Tensor *GetInputTensor(size_t index) const
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| index | 输入 | 输入index。 |

## 返回值说明

Tensor指针，异常时返回空指针。

## 约束说明

无

## 调用示例

以下片段位于`HostCpuExecuteOp::Execute`实现中，获取第0个输入Tensor。

```c++
ge::graphStatus Execute(gert::HostCpuOpExecutionContext *ctx) override {
  const gert::Tensor *input = ctx->GetInputTensor(0U);
  if (input == nullptr) {
    return ge::GRAPH_FAILED;
  }

  // 使用input获取输入Tensor的描述信息或数据。
  return ge::GRAPH_SUCCESS;
}
```
