# append\_workspace

## 产品支持情况

全量芯片支持。

## 头文件/库文件

- 头文件：无
- 库文件：ge_custom_op_native.so、libge_python_custom_op_bridge.so

## 功能说明

向当前kernel参数序列追加一个workspace地址参数。追加位置由调用顺序决定。

## 函数原型

```python
append_workspace(workspace: WorkspaceAddr) -> None
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| :--- | :--- | :--- |
| workspace | 输入 | 由`AnnotatedArgsContext.malloc_workspace()`返回的workspace地址信息。 |

## 返回值说明

无

## 约束说明

- `AnnotatedKernelArgs`只能由当前`AnnotatedArgsContext`的`create_kernel_args()`创建，并且只能在当前`declare_launch_args`回调内使用。
- `WorkspaceAddr`是借用对象，只能在获取它的`declare_launch_args`回调内使用。
- 当前对象已过期或已被`add_launch`消费时，调用此方法抛出`RuntimeError`。
