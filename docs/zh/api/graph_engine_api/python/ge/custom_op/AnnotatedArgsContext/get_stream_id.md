# get\_stream\_id

## 产品支持情况

全量芯片支持。

## 功能说明

获取当前`AnnotatedArgsContext`的stream标识。创建`AnnotatedKernelLaunchInfo`时可使用该标识设置`stream_id`。

## 函数原型

```python
get_stream_id() -> int
```

## 参数说明

无

## 返回值说明

| 类型 | 说明 |
| :--- | :--- |
| int | 当前`AnnotatedArgsContext`的stream标识。 |

## 调用示例

```python
from ge.custom_op import AnnotatedKernelLaunchInfo, get_declare_launch_args_ctx


def declare_launch_args(self, x1, x2, y) -> None:
    ctx = get_declare_launch_args_ctx()
    stream_id = ctx.get_stream_id()
    kernel_bin = b"..."
    launch_info = AnnotatedKernelLaunchInfo(
        kernel_name="add_custom",
        kernel_bin=kernel_bin,
        block_dim=8,
        stream_id=stream_id,
    )
```

## 约束说明

- 此方法只能在当前`declare_launch_args`回调内调用。
- 当前`AnnotatedArgsContext`已过期时，抛出`RuntimeError`。
