# append\_workspace

## 产品支持情况

全量芯片支持。

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

- `AnnotatedKernelArgs`只能由当前`AnnotatedArgsContext`的`create_kernel_args()`创建，并且只能在当前`declare_launch_args`回调内使用。
- `WorkspaceAddr`是借用对象，只能在获取它的`declare_launch_args`回调内使用。
- 当前对象已过期或已被`add_launch`消费时，调用此方法抛出`RuntimeError`。
