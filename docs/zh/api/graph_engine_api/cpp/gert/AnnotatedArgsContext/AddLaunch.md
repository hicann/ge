# AddLaunch

## 产品支持情况

全量芯片支持。

## 头文件/库文件

- 头文件：\#include <exe\_graph/runtime/annotated\_args\_context.h\>
- 库文件：liblowering.so

## 功能说明

添加一个声明式kernel launch。接口提取[`AnnotatedKernelArgs`](../AnnotatedKernelArgs/overview.md)的参数，并记录kernel名称、二进制、block dim和逻辑stream ID。

第二个原型在添加launch的同时返回其`AnnotatedLaunchToken`，并可声明依赖的前置launch token，用于表达多次launch之间的先后关系。依赖关系通过`AnnotatedLaunchToken`表达：每个经第二个原型成功提交的launch获得一个token，后续launch可引用这些token作为前驱。

<!-- npu="x90,9030" id1 -->
端侧场景：要求恰好调用一次。
<!-- end id1 -->

## 函数原型

```c++
ge::graphStatus AddLaunch(const AnnotatedKernelLaunchInfo &launch_info, AnnotatedKernelArgs &&args)
AnnotatedLaunchToken AddLaunch(const AnnotatedKernelLaunchInfo &launch_info, AnnotatedKernelArgs &&args,
                               const std::vector<AnnotatedLaunchToken> &predecessors)
```

- 第一个原型：添加一个kernel launch，不产生token，也不能作为其他launch的依赖源。
- 第二个原型：添加一个kernel launch并返回其token；`predecessors`声明本launch依赖的前驱token列表，无依赖时传空vector（`{}`）。

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| launch_info | 输入 | kernel launch信息。`kernel_name`和`kernel_bin`必须非空，`kernel_bin_size`和`block_dim`必须大于0；`stream_id`应使用[`GetStreamId`](GetStreamId.md)或[`RequestAttachedStream`](RequestAttachedStream.md)返回的合法逻辑辅流ID。详见[`AnnotatedKernelLaunchInfo`](../AnnotatedKernelLaunchInfo.md)。接口在调用期间GE会复制并保存`kernel_name`和`kernel_bin`数据；调用方可在接口返回后结束这些临时数据的生命周期。 |
| args | 输入 | kernel launch参数构建器，以右值引用移交。参数必须非空，且此前的参数追加操作均成功。调用期间GE会复制并保存args数据；调用方可在接口返回后结束args的生命周期。调用方以`std::move`移交后，不应依赖该对象的后续状态。 |
| predecessors | 输入 | 前驱launch的token列表，可为空。仅在第二个原型中使用。元素必须是本Context内此前第二个原型返回的有效token，不可重复、不可前向依赖；列表内容在调用期间复制，调用方可在接口返回后结束其生命周期。 |

## 返回值说明

- 第一个原型：

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| - | ge::graphStatus | `GRAPH_SUCCESS(0)`：添加成功；其他值：参数或Context状态异常，添加失败。 |

- 第二个原型：

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| - | AnnotatedLaunchToken | 成功时返回本次launch的新token（`uint32_t`，从0递增），供后续launch声明依赖；失败时返回UINT32\_MAX，且不发布launch。 |

## 约束说明

- 调用方必须检查每次调用的返回值，并在失败时立即返回该错误状态。

## 调用示例

以下`kKernelBin`和`kKernelBinSize`仅表示用户实际编译得到的kernel二进制及其大小，并非可执行的示例二进制。

- 第一个原型：

```c++
extern const uint8_t kKernelBin[];
extern const size_t kKernelBinSize;

const auto *input = ctx.GetInputTensor(0U);
const auto *output = ctx.GetOutputTensor(0U);
if ((input == nullptr) || (output == nullptr)) {
  return ge::GRAPH_FAILED;
}

gert::AnnotatedKernelArgs args(
    gert::InputAddr{0U, input->GetAddr()},
    gert::OutputAddr{0U, output->GetAddr()},
    uint64_t{1U});
const auto ret = ctx.AddLaunch(
    gert::AnnotatedKernelLaunchInfo{
        "my_kernel", kKernelBin, kKernelBinSize, 32U, ctx.GetStreamId()},
    std::move(args));
if (ret != ge::GRAPH_SUCCESS) {
  return ret;
}
return ge::GRAPH_SUCCESS;
```

- 第二个原型（`args0`和`args1`为已构造完成的[`AnnotatedKernelArgs`](../AnnotatedKernelArgs/overview.md)）：

```c++
extern const uint8_t kKernelBin[];
extern const size_t kKernelBinSize;

// 在主流上提交首个launch，无前驱，获取token。
const auto token0 = ctx.AddLaunch(
    gert::AnnotatedKernelLaunchInfo{"main_kernel", kKernelBin, kKernelBinSize, 32U, ctx.GetStreamId()},
    std::move(args0), {});
if (token0 == UINT32_MAX) {
  return ge::GRAPH_FAILED;
}

// 申请逻辑辅流，并声明辅流launch依赖主流launch。
const auto aux_stream_id = ctx.RequestAttachedStream(ge::AscendString("aux"));
if (aux_stream_id == UINT32_MAX) {
  return ge::GRAPH_FAILED;
}
const auto token1 = ctx.AddLaunch(
    gert::AnnotatedKernelLaunchInfo{"aux_kernel", kKernelBin, kKernelBinSize, 32U, aux_stream_id},
    std::move(args1), {token0});
if (token1 == UINT32_MAX) {
  return ge::GRAPH_FAILED;
}
return ge::GRAPH_SUCCESS;
```
