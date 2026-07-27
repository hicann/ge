# DeclareLaunchArgs

## 产品支持情况

全量芯片支持。

## 头文件/库文件

- 头文件：\#include <graph/custom\_op.h\>
- 库文件：libgraph.so

## 功能说明

在静态图编译阶段声明自定义算子的kernel launch参数。算子可以通过`ctx`获取输入/输出Tensor、申请逻辑workspace，构造[`gert::AnnotatedKernelArgs`](../../gert/AnnotatedKernelArgs/overview.md)，并使用[`gert::AnnotatedArgsContext::AddLaunch`](../../gert/AnnotatedArgsContext/AddLaunch.md)提交kernel launch。

## 函数原型

```c++
virtual graphStatus DeclareLaunchArgs(gert::AnnotatedArgsContext &ctx) = 0
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| ctx | 输入 | 声明式参数上下文。可通过上下文获取输入/输出Tensor、申请workspace、获取节点逻辑stream，并添加kernel launch。详见[`gert::AnnotatedArgsContext`](../../gert/AnnotatedArgsContext/overview.md)。 |

## 返回值说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| - | graphStatus | `GRAPH_SUCCESS(0)`：声明成功；其他值：声明失败，GE将终止当前自定义算子的任务生成。 |

## 约束说明

- 回调成功返回前，必须至少调用一次`AddLaunch`。
- 调用方必须检查每次`AddLaunch`的返回值；若返回非`GRAPH_SUCCESS`，应立即将该状态返回，使本接口返回失败。
<!-- npu="x90,9030" id1 -->
- 端侧场景：只允许调用一次`AddLaunch`。
<!-- end id1 -->

## 调用示例

```c++
#include <cstdint>
#include <utility>

#include "graph/custom_op.h"

class MyAnnotatedOp : public ge::AnnotatedArgsOp {
 public:
  ge::graphStatus DeclareLaunchArgs(gert::AnnotatedArgsContext &ctx) override {
    // 仅为文档占位；实际使用时必须替换为已编译且与kernel_name匹配的有效kernel二进制。
    static const uint8_t kKernelBin[] = {0x01U, 0x02U, 0x03U, 0x04U};

    const auto *input = ctx.GetInputTensor(0U);
    const auto *output = ctx.GetOutputTensor(0U);
    if ((input == nullptr) || (output == nullptr)) {
      return ge::GRAPH_FAILED;
    }

    const auto workspace = ctx.MallocWorkSpace(1024U);
    if (workspace.addr == nullptr) {
      return ge::GRAPH_FAILED;
    }

    gert::AnnotatedKernelArgs args(
        gert::InputAddr{0U, input->GetAddr()},
        gert::OutputAddr{0U, output->GetAddr()},
        workspace,
        uint64_t{1U});
    const auto launch_ret = ctx.AddLaunch(
        gert::AnnotatedKernelLaunchInfo{
            "my_annotated_kernel", kKernelBin, sizeof(kKernelBin), 32U, ctx.GetStreamId()},
        std::move(args));
    if (launch_ret != ge::GRAPH_SUCCESS) {
      return launch_ret;
    }
    return ge::GRAPH_SUCCESS;
  }
};
```
上述示例使用`ctx.GetStreamId()`保证launch与节点主stream一致，并将参数构建器以`std::move`交给`AddLaunch`。
