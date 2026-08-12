# add\_launch

## 产品支持情况

全量芯片支持。

## 头文件/库文件

- 头文件：无
- 库文件：ge_custom_op_native.so、libge_python_custom_op_bridge.so

## 功能说明

提交一个kernel launch的元数据和参数声明。调用后，GE记录该launch的参数顺序，用于静态模型的地址刷新。

## 函数原型

```python
add_launch(launch_info: AnnotatedKernelLaunchInfo, args: AnnotatedKernelArgs) -> None
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| :--- | :--- | :--- |
| launch_info | 输入 | kernel launch元数据，类型为`AnnotatedKernelLaunchInfo`，包含kernel名称、二进制、block数和stream标识。 |
| args | 输入 | kernel参数构造器，类型为`AnnotatedKernelArgs`。 |

## 返回值说明

无

## 约束说明

- 此方法只能在当前`declare_launch_args`回调内调用。
- `args`必须由当前`AnnotatedArgsContext`通过`create_kernel_args`创建。`launch_info.stream_id`校验通过后才会消费`args`；此后即使底层`AddLaunch`失败也不得复用。
- `launch_info.stream_id`必须等于当前`AnnotatedArgsContext`的stream标识。不匹配时会在消费`args`前抛出`ValueError`，此时builder尚未被消费，可以继续使用。
- 底层添加kernel launch失败，或当前`AnnotatedArgsContext`、`args`已过期时，抛出`RuntimeError`。
