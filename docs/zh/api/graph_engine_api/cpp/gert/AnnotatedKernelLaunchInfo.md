# AnnotatedKernelLaunchInfo

## 产品支持情况

全量芯片支持。

## 头文件/库文件

- 头文件：\#include <exe\_graph/runtime/annotated\_args\_context.h\>
- 库文件：liblowering.so

## 功能说明

`gert::AnnotatedKernelLaunchInfo`描述一次声明式kernel launch的元信息，作为[`AnnotatedArgsContext::AddLaunch`](AnnotatedArgsContext/AddLaunch.md)的第一个参数。它只描述kernel入口、二进制、block维度和逻辑stream；kernel参数本身由[`AnnotatedKernelArgs`](AnnotatedKernelArgs/overview.md)提供。

## 函数原型

```c++
namespace gert {
struct AnnotatedKernelLaunchInfo {
  const char *kernel_name = nullptr;  // kernel二进制中的入口函数名称
  const void *kernel_bin = nullptr;   // kernel二进制数据
  size_t kernel_bin_size = 0U;        // kernel二进制数据大小，单位为字节
  uint32_t block_dim = 0U;            // kernel的block维度
  uint32_t stream_id = 0U;            // launch所在的逻辑stream ID
};
}  // namespace gert
```

## 参数说明

| 参数名 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `kernel_name` | `const char *` | `nullptr` | kernel二进制中的入口函数名称。传给`AddLaunch`时必须为非空且首字符不能是`\0`。 |
| `kernel_bin` | `const void *` | `nullptr` | kernel二进制数据起始地址。传给`AddLaunch`时必须非空。 |
| `kernel_bin_size` | `size_t` | `0U` | 二进制数据大小，单位为字节；必须大于0。 |
| `block_dim` | `uint32_t` | `0U` | kernel的block维度；必须大于0。 |
| `stream_id` | `uint32_t` | `0U` | launch所在的逻辑stream ID。应使用[`AnnotatedArgsContext::GetStreamId`](AnnotatedArgsContext/GetStreamId.md)的返回值，并与节点主stream一致。 |

## 返回值说明

不涉及。

## 约束说明

无

## 调用示例

以下`kKernelBin`和`kKernelBinSize`仅表示用户实际编译得到的kernel二进制及其大小，并非可执行的示例二进制。

```c++
extern const uint8_t kKernelBin[];
extern const size_t kKernelBinSize;

gert::AnnotatedKernelArgs args(
    gert::InputAddr{0U, input_addr}, uint64_t{1U});
const gert::AnnotatedKernelLaunchInfo launch_info{
    "my_kernel", kKernelBin, kKernelBinSize, 32U, ctx.GetStreamId()};
const auto ret = ctx.AddLaunch(launch_info, std::move(args));
if (ret != ge::GRAPH_SUCCESS) {
  return ret;
}
return ge::GRAPH_SUCCESS;
```
