# Execute

## 产品支持情况

全量芯片支持。

## 头文件/库文件

- 头文件：\#include <graph/custom\_op.h\>
- 库文件：liblowering.so

## 功能说明

Host CPU自定义算子的执行函数。

## 函数原型

```c++
virtual graphStatus Execute(gert::HostCpuOpExecutionContext *ctx) = 0
```

## 参数说明

| 参数名 | 输入/输出 | 说明                                                         |
| --- | --- |--------------------------------------------------------------|
| ctx | 输入 | 执行时上下文，可通过上下文获取input tensor，分配输出内存等。 |

## 返回值说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| - | graphStatus | `GRAPH_SUCCESS(0)`：执行成功；其他值：执行失败。 |

## 约束说明

无

## 调用示例

以下示例在自定义算子的Execute中读取两个输入，申请一个输出并完成Host CPU计算。

```c++
#include "graph/custom_op.h"

class AddHostCpu final : public ge::HostCpuExecuteOp {
 public:
  ge::graphStatus Execute(gert::HostCpuOpExecutionContext *ctx) override {
    const auto *x = ctx->GetInputTensor(0U);
    const auto *y = ctx->GetInputTensor(1U);
    if ((x == nullptr) || (y == nullptr)) {
      return ge::GRAPH_FAILED;
    }

    auto *z = ctx->MallocOutputTensor(0U, x->GetShape(), x->GetFormat(), x->GetDataType());
    if (z == nullptr) {
      return ge::GRAPH_FAILED;
    }

    // 根据x、y完成Host CPU计算。
    return ge::GRAPH_SUCCESS;
  }
};

REG_OP_BACKEND(AddHostCpu, "Add", ge::OpBackend::kHostCPU);
```
