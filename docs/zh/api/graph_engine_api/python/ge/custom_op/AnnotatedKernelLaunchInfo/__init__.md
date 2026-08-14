# \_\_init\_\_

## 产品支持情况

全量芯片支持。

## 功能说明

创建一个kernel launch的元数据对象，用于向`AnnotatedArgsContext.add_launch`提交kernel名称、二进制、block数和stream标识。

## 函数原型

```python
__init__(*, kernel_name: str, kernel_bin: bytes, block_dim: int, stream_id: int) -> None
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| :--- | :--- | :--- |
| kernel_name | 输入 | kernel名称。不能为空字符串。 |
| kernel_bin | 输入 | kernel二进制。不能为空字节串。 |
| block_dim | 输入 | kernel的block数。取值范围为1到2^32-1。 |
| stream_id | 输入 | kernel所在stream的标识。取值范围为0到2^32-1。 |

## 返回值说明

无

## 调用示例

`kernel_bin`示例中表示已经准备好的kernel二进制字节串。

```python
from ge.custom_op import AnnotatedKernelLaunchInfo, get_declare_launch_args_ctx


def declare_launch_args(self, x1, x2, y) -> None:
    ctx = get_declare_launch_args_ctx()
    kernel_bin = b"..."
    launch_info = AnnotatedKernelLaunchInfo(
        kernel_name="add_custom",
        kernel_bin=kernel_bin,
        block_dim=8,
        stream_id=ctx.get_stream_id(),
    )
```

## 约束说明

- 参数必须使用关键字方式传入。
- 构造时会复制`kernel_name`和`kernel_bin`，此对象独立持有这两份数据。
- `kernel_name`为空、`kernel_bin`为空或`block_dim`为0时，抛出`ValueError`。
- 提交时，`stream_id`必须与当前`AnnotatedArgsContext`的stream标识相同；不匹配时，`AnnotatedArgsContext.add_launch`抛出`ValueError`。
