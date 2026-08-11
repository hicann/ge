# 简介

`AnnotatedKernelLaunchInfo`用于描述一个kernel launch的名称、二进制、block数和stream标识。调用[AnnotatedArgsContext.add_launch](../AnnotatedArgsContext/add_launch.md)时，将该对象与`AnnotatedKernelArgs`一同提交。

该对象由Python创建。构造时会复制kernel名称和二进制，并独立持有这两份数据。创建后可在当前`declare_launch_args`回调外保存，但提交时的`stream_id`必须与当前`AnnotatedArgsContext`的stream标识相同。
