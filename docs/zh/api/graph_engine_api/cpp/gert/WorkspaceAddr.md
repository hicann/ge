# WorkspaceAddr

## 产品支持情况

全量芯片支持。

## 头文件/库文件

- 头文件：\#include <exe\_graph/runtime/annotated\_args\_context.h\>
- 库文件：liblowering.so

## 功能说明

`gert::WorkspaceAddr`是逻辑workspace地址描述符，作为[`AnnotatedKernelArgs::AppendArg`](AnnotatedKernelArgs/AppendArg.md)的参数，用于把kernel args中的一个参数标记为workspace地址。

## 函数原型

```c++
namespace gert {
struct WorkspaceAddr {
  uint32_t index;    // workspace序号，由MallocWorkSpace返回
  const void *addr;  // workspace逻辑地址
};
}  // namespace gert
```

## 参数说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| `index` | `uint32_t` | workspace序号，由[`AnnotatedArgsContext::MallocWorkSpace`](AnnotatedArgsContext/MallocWorkSpace.md)返回。序号按workspace申请顺序从0递增，不是算子IR输入/输出index；值必须不大于`INT32_MAX`。 |
| `addr` | `const void *` | 对应workspace的逻辑地址，来自`MallocWorkSpace`返回结构体中的`addr`。|

## 返回值说明

不涉及。

## 约束说明

无

## 调用示例

```c++
const auto workspace = ctx.MallocWorkSpace(workspace_size);
if (workspace.addr == nullptr) {
  return ge::GRAPH_FAILED;
}
gert::AnnotatedKernelArgs args(workspace);
return ctx.AddLaunch(launch_info, std::move(args));
```
