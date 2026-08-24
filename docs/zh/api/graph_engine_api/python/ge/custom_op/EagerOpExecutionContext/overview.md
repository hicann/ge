# 简介

`EagerOpExecutionContext`是Python自定义算子`execute`使用的运行时执行上下文。用户通过[get_execute_ctx](../get_execute_ctx.md)获取该对象，不能直接构造。

该对象是仅在当前`execute`内有效的借用视图。通过该对象可以查询输出`Tensor`、申请输出和workspace内存以及获取执行流句柄。`execute`执行结束后，上下文及其返回的借用对象都会失效。
