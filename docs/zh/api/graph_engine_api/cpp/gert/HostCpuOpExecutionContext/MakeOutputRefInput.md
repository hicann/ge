# MakeOutputRefInput

## 产品支持情况

全量芯片支持。

## 头文件/库文件

- 头文件：\#include <exe\_graph/runtime/host\_cpu\_op\_execution\_context.h\>
- 库文件：liblowering.so

## 功能说明

指定某输出的内存地址引用自某个输入。

## 函数原型

```c++
Tensor *MakeOutputRefInput(size_t output_index, size_t input_index)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| output_index | 输入 | 输出索引。 |
| input_index | 输入 | 输入索引。 |

## 返回值说明

output\_index对应的输出Tensor指针。

## 约束说明

- output_index对应的输出参数和input_index对应的输入参数，在算子IR原型定义中的名称必须一致，否则接口调用失败。

## 调用示例

以下为一个简单的算子IR原型定义，输入和输出参数的名称均为x。

```c++
REG_OP(Identity)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(x, TensorType::ALL())
    .OP_END_FACTORY_REG(Identity)
```

以下片段位于`HostCpuExecuteOp::Execute`实现中，将第0个输出设置为引用第0个输入的内存地址，并检查返回值。

```c++
ge::graphStatus Execute(gert::HostCpuOpExecutionContext *ctx) override {
  gert::Tensor *output = ctx->MakeOutputRefInput(0U, 0U);
  if (output == nullptr) {
    return ge::GRAPH_FAILED;
  }

  return ge::GRAPH_SUCCESS;
}
```
