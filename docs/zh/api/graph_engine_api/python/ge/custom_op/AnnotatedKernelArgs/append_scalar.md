# append\_scalar

## 产品支持情况

全量芯片支持。

## 头文件/库文件

- 头文件：无
- 库文件：ge_custom_op_native.so、libge_python_custom_op_bridge.so

## 功能说明

向当前kernel参数序列追加一个64位无符号标量参数。追加位置由调用顺序决定。

## 函数原型

```python
append_scalar(value: int) -> None
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| :--- | :--- | :--- |
| value | 输入 | 64位无符号标量值，取值范围为0到2^64-1。 |

## 返回值说明

无

## 约束说明

- `AnnotatedKernelArgs`只能由当前`AnnotatedArgsContext`的`create_kernel_args()`创建，并且只能在当前`declare_launch_args`回调内使用。
- 当前对象已过期或已被`add_launch`消费时，调用此方法抛出`RuntimeError`。
