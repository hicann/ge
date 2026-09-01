# 简介

`ge.runtime`提供Python自定义算子执行和元信息推导使用的运行时数据结构。`TensorDesc`用于`register_op`装饰函数的输入和返回值；`Tensor`、`StorageShape`、`StorageFormat`等对象由运行时context在回调期间提供。

其中`ge.runtime.TensorDesc`与图构建API中的[`ge.graph.TensorDesc`](../graph/TensorDesc/overview.md)是不同类型，使用场景和接口不可混用。

- [`Tensor`](Tensor/overview.md)：执行回调期间使用的张量运行时视图。
- [`TensorDesc`](TensorDesc/overview.md)：Python自定义算子`infer_meta`使用的张量元信息。
