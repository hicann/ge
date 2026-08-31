# GetOptionalInputTensor

## 产品支持情况

全量芯片支持。

## 头文件/库文件

- 头文件：\#include <exe\_graph/runtime/host\_cpu\_op\_execution\_context.h\>
- 库文件：liblowering.so

## 功能说明

基于算子IR原型定义，获取OPTIONAL_INPUT类型的输入tensor指针。

## 函数原型

```c++
const Tensor *GetOptionalInputTensor(size_t ir_index) const
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| ir_index | 输入 | IR原型定义中的index。 |

## 返回值说明

Tensor指针，异常时返回空指针。

## 约束说明

无

## 调用示例

以下片段位于`HostCpuExecuteOp::Execute`实现中，获取算子IR原型中index为1的`OPTIONAL_INPUT`输入。未提供可选输入时，返回空指针。

```c++
ge::graphStatus Execute(gert::HostCpuOpExecutionContext *ctx) override {
  const gert::Tensor *optional_input = ctx->GetOptionalInputTensor(1U);
  if (optional_input != nullptr) {
    // 使用optional_input完成可选输入相关的计算。
  }

  return ge::GRAPH_SUCCESS;
}
```
