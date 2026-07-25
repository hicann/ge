# AddLaunch

## 产品支持情况

全量芯片支持。

## 头文件/库文件

- 头文件：\#include <exe\_graph/runtime/annotated\_args\_context.h\>
- 库文件：liblowering.so

## 功能说明

添加一个声明式kernel launch。接口提取[`AnnotatedKernelArgs`](../AnnotatedKernelArgs/overview.md)的参数，并记录kernel名称、二进制、block dim和逻辑stream ID。

<!-- npu="x90,9030" id1 -->
端侧场景：要求恰好调用一次。
<!-- end id1 -->

## 函数原型

```c++
ge::graphStatus AddLaunch(const AnnotatedKernelLaunchInfo &launch_info, AnnotatedKernelArgs &&args)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| launch_info | 输入 | kernel launch信息。`kernel_name`和`kernel_bin`必须非空，`kernel_bin_size`和`block_dim`必须大于0；`stream_id`应使用[`GetStreamId`](GetStreamId.md)的返回值。详见[`AnnotatedKernelLaunchInfo`](../AnnotatedKernelLaunchInfo.md)。接口在调用期间GE会复制并保存`kernel_name`和`kernel_bin`数据；调用方可在接口返回后结束这些临时数据的生命周期。 |
| args | 输入 | kernel launch参数构建器，以右值引用移交。参数必须非空，且此前的参数追加操作均成功。调用期间GE会复制并保存args数据；调用方可在接口返回后结束args的生命周期。调用方以`std::move`移交后，不应依赖该对象的后续状态。 |

## 返回值说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| - | ge::graphStatus | `GRAPH_SUCCESS(0)`：添加成功；其他值：参数或Context状态异常，添加失败。 |

## 约束说明

- 调用方必须检查每次调用的返回值，并在失败时立即返回该错误状态。

## 调用示例

以下`kKernelBin`和`kKernelBinSize`仅表示用户实际编译得到的kernel二进制及其大小，并非可执行的示例二进制。

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
