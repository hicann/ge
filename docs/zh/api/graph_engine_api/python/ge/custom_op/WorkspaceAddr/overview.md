# 简介

`WorkspaceAddr`描述由[AnnotatedArgsContext.malloc_workspace](../AnnotatedArgsContext/malloc_workspace.md)申请的workspace。该对象可传入`AnnotatedKernelArgs.append_workspace`，以声明kernel的workspace地址参数。

`WorkspaceAddr`不能直接构造，只能由`malloc_workspace()`返回。它是借用对象，只在当前`declare_launch_args`回调内有效；回调返回或抛出异常后，随`AnnotatedArgsContext`一同失效，再次访问会抛出`RuntimeError`。
