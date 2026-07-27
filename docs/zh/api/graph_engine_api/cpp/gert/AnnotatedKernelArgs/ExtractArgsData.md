# ExtractArgsData

## 产品支持情况

全量芯片支持。

## 头文件/库文件

- 头文件：\#include <exe\_graph/runtime/annotated\_args\_context.h\>
- 库文件：liblowering.so

## 功能说明

提取`gert::AnnotatedKernelArgs`已构建的kernel参数数据和参数描述。自定义算子开发者可用它检查构建结果，但不应自行重排或改写描述。

## 函数原型

```c++
ge::graphStatus ExtractArgsData(std::vector<uint8_t> &args_data,
                                std::vector<ge::ArgDesc> &arg_descs) const
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| `args_data` | 输出 | kernel args的字节数据。数据顺序与追加顺序一致。 |
| `arg_descs` | 输出 | 与`args_data`中参数一一对应的`ge::ArgDesc`描述数组。地址参数描述输入、输出或workspace的类型和index；`uint64_t`参数描述自定义值。 |

调用成功时，两个vector的原有内容会被结果覆盖；调用失败时不要使用输出内容。

## 返回值说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| - | `ge::graphStatus` | `GRAPH_SUCCESS(0)`：提取成功；其他值：对象为空或状态无效、此前追加失败、没有参数，或参数数据与描述数量不一致。 |

## 约束说明

- 必须至少成功追加一个参数；空的`AnnotatedKernelArgs`提取失败。
- 所有追加操作必须成功，且`args_data.size()`必须等于`arg_descs.size() * sizeof(uint64_t)`；任一失败状态都会使提取失败。
- `args_data[i]`所在的第`i`个参数与`arg_descs[i]`严格对应。

## 调用示例

```c++
gert::AnnotatedKernelArgs args(
    gert::InputAddr{0U, input_addr}, uint64_t{7U});
std::vector<uint8_t> args_data;
std::vector<ge::ArgDesc> arg_descs;
if (args.ExtractArgsData(args_data, arg_descs) != ge::GRAPH_SUCCESS) {
  return ge::GRAPH_FAILED;
}

// 两个参数各占8字节，描述和数据按同一顺序对应。
if ((args_data.size() != 2U * sizeof(uint64_t)) ||
    (arg_descs.size() != 2U)) {
  return ge::GRAPH_FAILED;
}
return ctx.AddLaunch(launch_info, std::move(args));
```
