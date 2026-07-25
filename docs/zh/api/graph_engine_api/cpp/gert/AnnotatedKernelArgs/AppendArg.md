# AppendArg

## 产品支持情况

全量芯片支持。

## 头文件/库文件

- 头文件：\#include <exe\_graph/runtime/annotated\_args\_context.h\>
- 库文件：liblowering.so

## 功能说明

向`gert::AnnotatedKernelArgs`末尾追加一个参数。参数的追加顺序就是kernel args中的排列顺序；每次成功追加占用一个8-byte参数，并同步生成一个用于地址刷新或标量透传的`ge::ArgDesc`。

## 函数原型

```c++
ge::graphStatus AppendArg(const InputAddr &addr)
ge::graphStatus AppendArg(const OutputAddr &addr)
ge::graphStatus AppendArg(const WorkspaceAddr &addr)
ge::graphStatus AppendArg(uint64_t value)
```

## 参数说明

| 函数 | 参数名 | 输入/输出 | 说明 |
| --- | --- | --- | --- |
| `AppendArg(const InputAddr &)` | `addr` | 输入 | 逻辑输入地址描述符。详见[`InputAddr`](../InputAddr.md)。 |
| `AppendArg(const OutputAddr &)` | `addr` | 输入 | 逻辑输出地址描述符。详见[`OutputAddr`](../OutputAddr.md)。 |
| `AppendArg(const WorkspaceAddr &)` | `addr` | 输入 | 逻辑workspace地址描述符。详见[`WorkspaceAddr`](../WorkspaceAddr.md)。 |
| `AppendArg(uint64_t)` | `value` | 输入 | 直接写入kernel args的64-bit标量值，并生成自定义值描述。 |

## 返回值说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| - | `ge::graphStatus` | `GRAPH_SUCCESS(0)`：追加成功；其他值：对象无效、IR index超出范围或内部参数存储失败。 |

## 约束说明

- 追加失败会使构建器状态失败；后续不要把该对象传给`ExtractArgsData`或`AddLaunch`。

## 调用示例

```c++
const auto *input = ctx.GetInputTensor(0U);
const auto *output = ctx.GetOutputTensor(0U);
const auto workspace = ctx.MallocWorkSpace(workspace_size);
if ((input == nullptr) || (output == nullptr) || (workspace.addr == nullptr)) {
  return ge::GRAPH_FAILED;
}

gert::AnnotatedKernelArgs args;
if (args.AppendArg(gert::InputAddr{0U, input->GetAddr()}) != ge::GRAPH_SUCCESS ||
    args.AppendArg(gert::OutputAddr{0U, output->GetAddr()}) != ge::GRAPH_SUCCESS ||
    args.AppendArg(gert::WorkspaceAddr{workspace.index, workspace.addr}) != ge::GRAPH_SUCCESS ||
    args.AppendArg(uint64_t{1U}) != ge::GRAPH_SUCCESS) {
  return ge::GRAPH_FAILED;
}
// args按上述顺序占用4个参数。
return ctx.AddLaunch(launch_info, std::move(args));
```
