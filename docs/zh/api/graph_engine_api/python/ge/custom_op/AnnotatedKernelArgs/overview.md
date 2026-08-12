# 简介

`AnnotatedKernelArgs`用于声明单个kernel launch的参数顺序。只能由[AnnotatedArgsContext.create_kernel_args](../AnnotatedArgsContext/create_kernel_args.md)创建，不能直接构造。

每次调用`append_input`、`append_output`、`append_workspace`或`append_scalar`都会在末尾追加一个参数，调用先后顺序就是kernel launch的参数顺序；这四类方法没有固定的调用次序。完成追加后，将对象传入[AnnotatedArgsContext.add_launch](../AnnotatedArgsContext/add_launch.md)提交。

该对象是借用对象，只能在当前`declare_launch_args`回调内使用。回调结束后对象失效；调用`add_launch`后对象被消费，二者都会导致后续访问抛出`RuntimeError`。
