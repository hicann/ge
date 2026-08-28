# GetDynamicInputTensor

## 产品支持情况

全量芯片支持。

## 头文件/库文件

- 头文件：\#include <exe\_graph/runtime/host\_cpu\_op\_execution\_context.h\>
- 库文件：liblowering.so

## 功能说明

根据算子IR原型定义，获取DYNAMIC_INPUT类型的输入Tensor指针。

## 函数原型

```c++
const Tensor *GetDynamicInputTensor(size_t ir_index, size_t relative_index) const
```

## 参数说明

| 参数名 | 输入/输出 | 说明                                |
| --- | --- |-------------------------------------|
| ir_index | 输入 | IR原型定义中的index。            |
| relative_index | 输入 | 该输入实例化后的相对index，例如某个DYNAMIC_INPUT实例化了3个输入，那么relative_index的有效范围是[0,2]。 |

## 返回值说明

Tensor指针，异常时返回空指针。

## 约束说明

无

## 调用示例

以下片段位于`HostCpuExecuteOp::Execute`实现中，获取算子IR原型中index为0的`DYNAMIC_INPUT`，其相对索引为1的实例化输入。

```c++
ge::graphStatus Execute(gert::HostCpuOpExecutionContext *ctx) override {
  const gert::Tensor *input = ctx->GetDynamicInputTensor(0U, 1U);
  if (input == nullptr) {
    return ge::GRAPH_FAILED;
  }

  // 使用input获取动态输入Tensor的描述信息或数据。
  return ge::GRAPH_SUCCESS;
}
```
