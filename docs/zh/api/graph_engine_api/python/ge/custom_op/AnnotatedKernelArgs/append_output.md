# append\_output

## 产品支持情况

全量芯片支持。

## 头文件/库文件

- 头文件：无
- 库文件：ge_custom_op_native.so、libge_python_custom_op_bridge.so

## 功能说明

向当前kernel参数序列追加一个输出张量参数。追加位置由调用顺序决定。

## 函数原型

```python
append_output(instance_index: int, tensor: Tensor) -> None
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| :--- | :--- | :--- |
| instance_index | 输入 | 当前计算节点输出的实例平铺索引。 |
| tensor | 输入 | 要追加的输出张量。调用者应传入当前回调中与该输出对应的tensor。 |

## 返回值说明

无

## 约束说明

- `AnnotatedKernelArgs`只能由当前`AnnotatedArgsContext`的`create_kernel_args()`创建，并且只能在当前`declare_launch_args`回调内使用。
- `instance_index`必须位于当前计算节点输出的实例平铺索引范围内。索引越界时抛出`IndexError`。
- 当前对象已过期或已被`add_launch`消费时，调用此方法抛出`RuntimeError`。
