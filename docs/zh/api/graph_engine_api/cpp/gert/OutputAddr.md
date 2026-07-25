# OutputAddr

## 产品支持情况

全量芯片支持。

## 头文件/库文件

- 头文件：\#include <exe\_graph/runtime/annotated\_args\_context.h\>
- 库文件：liblowering.so

## 功能说明

`gert::OutputAddr`是逻辑输出地址描述符，作为[`AnnotatedKernelArgs::AppendArg`](AnnotatedKernelArgs/AppendArg.md)的参数，用于把kernel args中的一个参数标记为算子输出地址。

## 函数原型

```c++
namespace gert {
struct OutputAddr {
  uint32_t index;    // 算子实例中的输出index
  const void *addr;  // 输出逻辑地址
};
}  // namespace gert
```

## 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `index` | `uint32_t` | 算子实例中的输出index。值必须不大于`INT32_MAX`。 |
| `addr` | `const void *` | 追加到args参数的输出逻辑地址。 |

## 返回值说明

无

## 约束说明

- `addr`应来自当前编译期回调可用的输出Tensor，例如[`AnnotatedArgsContext::GetOutputTensor`](AnnotatedArgsContext/GetOutputTensor.md)返回值的地址。

## 调用示例

```c++
const auto *output = ctx.GetOutputTensor(0U);
if (output == nullptr) {
  return ge::GRAPH_FAILED;
}
gert::AnnotatedKernelArgs args(
    gert::OutputAddr{0U, output->GetAddr()});
return ctx.AddLaunch(launch_info, std::move(args));
```
