# 简介

`gert::AnnotatedKernelArgs`是声明式kernel launch参数构建器。调用方按kernel参数布局依次追加逻辑输入地址、逻辑输出地址、逻辑workspace地址和`uint64_t`标量，再将对象以右值传给[`gert::AnnotatedArgsContext::AddLaunch`](../AnnotatedArgsContext/AddLaunch.md)。该接口仅用于[`ge::AnnotatedArgsOp::DeclareLaunchArgs`](../../ge/AnnotatedArgsOp/DeclareLaunchArgs.md)编译期回调构造静态图的launch参数。

## 需要包含的头文件

```c++
#include <exe_graph/runtime/annotated_args_context.h>
```

## Public成员函数

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
ge::graphStatus AppendArg(const InputAddr &addr)
ge::graphStatus AppendArg(const OutputAddr &addr)
ge::graphStatus AppendArg(const WorkspaceAddr &addr)
ge::graphStatus AppendArg(uint64_t value)
ge::graphStatus ExtractArgsData(std::vector<uint8_t> &args_data,
                                std::vector<ge::ArgDesc> &arg_descs) const
```

## 相关类型

- [`InputAddr`](../InputAddr.md)
- [`OutputAddr`](../OutputAddr.md)
- [`WorkspaceAddr`](../WorkspaceAddr.md)
- [`AnnotatedArgsContext`](../AnnotatedArgsContext/overview.md)
- [`AnnotatedKernelLaunchInfo`](../AnnotatedKernelLaunchInfo.md)

拷贝、移动和析构行为见[构造函数、赋值运算符和析构函数](AnnotatedKernelArgs_constructor_and_destructor.md)。
