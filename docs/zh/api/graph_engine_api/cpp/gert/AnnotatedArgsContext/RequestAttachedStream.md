# RequestAttachedStream

## 产品支持情况

全量芯片支持。

## 头文件/库文件

- 头文件：\#include <exe\_graph/runtime/annotated\_args\_context.h\>
- 库文件：liblowering.so

## 功能说明

按key申请逻辑辅流。key的作用域为算子所在的子图（对应一个编译子模型）：同一子图内相同key复用同一逻辑stream ID；不同子图之间key相互独立，即使key相同也各自分配stream ID，运行时对应各自独立的物理流。返回的逻辑stream ID可用于[`AnnotatedKernelLaunchInfo`](../AnnotatedKernelLaunchInfo.md)的`stream_id`字段，在辅流上提交kernel launch。

## 函数原型

```c++
uint32_t RequestAttachedStream(const ge::AscendString &key)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| key | 输入 | 辅流标识，作用域为算子所在的子图。不能为空，且不能使用`CANN-FMK-`保留前缀。 |

## 返回值说明

正常时返回key对应的逻辑辅流ID；key为空或申请失败时返回UINT32_MAX。

## 约束说明

- 返回的编译期逻辑ID只能用于当前Context的`AnnotatedKernelLaunchInfo::stream_id`。
- Context销毁后该ID不可作为运行时句柄使用。
- key不跨子图复用，本接口不支持在多个子图之间共享同一条物理辅流。

## 调用示例

```c++
const auto aux_stream_id = ctx.RequestAttachedStream(ge::AscendString("aux"));
if (aux_stream_id == UINT32_MAX) {
  return ge::GRAPH_FAILED;
}
// 将aux_stream_id填入AnnotatedKernelLaunchInfo::stream_id，
// 再调用带predecessors参数的[`AddLaunch`](AddLaunch.md)重载在辅流上提交launch。
```
