# get\_declare\_launch\_args\_ctx

## 产品支持情况

全量芯片支持。

## 功能说明

获取当前`declare_launch_args`回调的声明式参数上下文。通过返回的`AnnotatedArgsContext`可以申请逻辑workspace、获取stream标识、创建`AnnotatedKernelArgs`，并提交`AnnotatedKernelLaunchInfo`和kernel参数。

## 函数原型

```python
get_declare_launch_args_ctx() -> AnnotatedArgsContext
```

## 参数说明

无

## 返回值说明

| 类型 | 说明 |
| :--- | :--- |
| AnnotatedArgsContext | 当前`declare_launch_args`回调的借用上下文。 |

## 调用示例

```python
from ge.custom_op import get_declare_launch_args_ctx
from ge.runtime import Tensor


def declare_launch_args(self, x1: Tensor, x2: Tensor, y: Tensor) -> None:
    ctx = get_declare_launch_args_ctx()
    args = ctx.create_kernel_args()
```

## 约束说明

- 此接口只能在当前`declare_launch_args`回调内调用。回调外调用抛出`RuntimeError`。
- `AnnotatedArgsContext`是借用对象，仅在当前回调内有效。由其创建的`AnnotatedKernelArgs`和申请得到的`WorkspaceAddr`也只能在当前回调内使用。
- `AnnotatedKernelLaunchInfo`保存kernel名称、二进制、block数和stream标识。调用`AnnotatedArgsContext.add_launch`时会消费传入的`AnnotatedKernelArgs`。
- `AnnotatedKernelArgs.append_input`和`append_output`的`instance_index`分别使用当前计算节点输入、输出的实例平铺索引。
