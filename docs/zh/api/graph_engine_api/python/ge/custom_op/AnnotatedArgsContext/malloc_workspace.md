# malloc\_workspace

## 产品支持情况

全量芯片支持。

## 功能说明

为当前声明式参数上下文申请workspace，并返回该workspace的地址信息。返回的`WorkspaceAddr`可传入`AnnotatedKernelArgs.append_workspace`声明kernel参数。

## 函数原型

```python
malloc_workspace(size: int) -> WorkspaceAddr
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| :--- | :--- | :--- |
| size | 输入 | workspace大小，单位为字节。取值必须大于0。 |

## 返回值说明

| 类型 | 说明 |
| :--- | :--- |
| WorkspaceAddr | 当前回调申请的workspace地址信息，包含workspace索引和地址。 |

## 调用示例

```python
from ge.custom_op import get_declare_launch_args_ctx
from ge.runtime import Tensor


def declare_launch_args(self, x1: Tensor, x2: Tensor, y: Tensor) -> None:
    ctx = get_declare_launch_args_ctx()
    args = ctx.create_kernel_args()
    workspace = ctx.malloc_workspace(256)
    args.append_workspace(workspace)
```

## 约束说明

- 此方法只能在当前`declare_launch_args`回调内调用。`WorkspaceAddr`是借用对象，只能在当前回调内使用。
- workspace申请失败，或当前`AnnotatedArgsContext`已过期时，抛出`RuntimeError`。
