# AnnotatedKernelArgs构造函数、赋值运算符和析构函数

## 产品支持情况

全量芯片支持。

## 头文件/库文件

- 头文件：\#include <exe\_graph/runtime/annotated\_args\_context.h\>
- 库文件：liblowering.so

## 功能说明

构造、复制、移动或销毁`gert::AnnotatedKernelArgs`对象。对象内部拥有参数字节数据和参数描述的存储空间；拷贝操作复制这些数据，移动操作转移所有权。

## 函数原型

```c++
AnnotatedKernelArgs()
template <typename Arg, typename... Args,
          typename = typename std::enable_if<
              !std::is_same<typename std::decay<Arg>::type, AnnotatedKernelArgs>::value>::type>
explicit AnnotatedKernelArgs(Arg &&arg, Args &&...args)
AnnotatedKernelArgs(const AnnotatedKernelArgs &other)
AnnotatedKernelArgs(AnnotatedKernelArgs &&other) noexcept
AnnotatedKernelArgs &operator=(const AnnotatedKernelArgs &other)
AnnotatedKernelArgs &operator=(AnnotatedKernelArgs &&other) noexcept
~AnnotatedKernelArgs()
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| `arg` | 输入 | 变参构造函数中的第一个参数，类型必须为`InputAddr`、`OutputAddr`、`WorkspaceAddr`或`uint64_t`。 |
| `args` | 输入 | 变参构造函数中其余参数，按传入顺序追加。 |
| `other` | 输入 | 要复制或移动的源对象。 |

## 返回值说明

- 构造函数返回构造完成的`AnnotatedKernelArgs`对象。
- 拷贝赋值和移动赋值返回`*this`。
- 析构函数无返回值。

## 约束说明

以下约束同时规定对象的生命周期和移动语义。

- 默认构造函数创建空构建器；空构建器必须先成功追加至少一个参数，才能调用[`ExtractArgsData`](ExtractArgsData.md)或提交给[`AnnotatedArgsContext::AddLaunch`](../AnnotatedArgsContext/AddLaunch.md)。
- 变参构造函数按实参顺序调用`AppendArg`。如果某次追加失败，后续参数不会继续追加；该对象的状态不可作为有效launch参数使用，应检查后丢弃。
- 拷贝构造和拷贝赋值执行深拷贝：源、目标拥有独立的参数字节和描述数组，任一对象后续追加不会改变另一对象。地址字段复制的是地址值本身，不会复制地址指向的内存。
- 移动构造和移动赋值转移内部存储；移动完成后源对象不可再使用。对移动后的源对象调用任何`AppendArg`，或从中提取参数，均不能依赖成功结果。
- 自移动赋值（`obj = std::move(obj)`）保持对象可用；除该特殊情况外，不要在移动后继续访问源对象。
- 以右值将对象传给`AddLaunch`后，GE会从对象提取并复制launch所需数据；调用方不应依赖该对象在调用后的状态，也不要继续追加或提取参数。

## 调用示例

```c++
gert::AnnotatedKernelArgs original(
    gert::InputAddr{0U, input_addr}, uint64_t{1U});

// 拷贝后两个对象相互独立。
gert::AnnotatedKernelArgs copied(original);
(void)copied.AppendArg(gert::OutputAddr{0U, output_addr});

// moved获得copied的存储；copied移动后不可再使用。
gert::AnnotatedKernelArgs moved(std::move(copied));
const auto ret = ctx.AddLaunch(launch_info, std::move(moved));
if (ret != ge::GRAPH_SUCCESS) {
  return ret;
}
```

示例中的`input_addr`、`output_addr`和`launch_info`应由当前[`DeclareLaunchArgs`](../../ge/AnnotatedArgsOp/DeclareLaunchArgs.md)回调获得；`AddLaunch`的详细校验见[`AnnotatedKernelLaunchInfo`](../AnnotatedKernelLaunchInfo.md)。`AddLaunch`并不会因为右值引用而使`AnnotatedKernelArgs`的内部存储立即失效，但调用方仍不应依赖调用后的对象状态。
