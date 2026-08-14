# create\_kernel\_args

## 产品支持情况

全量芯片支持。

## 功能说明

创建一个用于声明单个kernel launch参数顺序的`AnnotatedKernelArgs`。可通过该对象追加输入、输出、workspace和标量参数，再传入`add_launch`提交声明。

## 函数原型

```python
create_kernel_args() -> AnnotatedKernelArgs
```

## 参数说明

无

## 返回值说明

| 类型 | 说明 |
| :--- | :--- |
| AnnotatedKernelArgs | 当前`AnnotatedArgsContext`创建的kernel参数构造器。 |

## 调用示例

```python
from ge.custom_op import get_declare_launch_args_ctx
from ge.runtime import Tensor


def declare_launch_args(self, x1: Tensor, x2: Tensor, y: Tensor) -> None:
    ctx = get_declare_launch_args_ctx()
    args = ctx.create_kernel_args()
    args.append_input(0, x1)
    args.append_input(1, x2)
    args.append_output(0, y)
```

## 约束说明

- 此方法只能在当前`declare_launch_args`回调内调用。返回的`AnnotatedKernelArgs`是借用对象，只能在当前回调内使用。
- 将返回的`AnnotatedKernelArgs`传入当前`AnnotatedArgsContext.add_launch`后，该构造器会被消费，不得再次使用。
- 当前`AnnotatedArgsContext`已过期时，抛出`RuntimeError`。
