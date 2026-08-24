# get\_execute\_ctx

## 产品支持情况

全量芯片支持。

## 功能说明

获取当前`execute`的运行时执行上下文。通过返回的`EagerOpExecutionContext`可以查询输出`Tensor`、申请输出和workspace内存，并获取执行流句柄。

## 函数原型

```python
get_execute_ctx() -> EagerOpExecutionContext
```

## 参数说明

无

## 返回值说明

| 类型 | 说明 |
| :--- | :--- |
| EagerOpExecutionContext | 当前`execute`的执行上下文。 |

## 调用示例

```python
from ge.custom_op import get_execute_ctx, register_op_impl
from ge.runtime import Tensor


@register_op_impl(op_type="AddPythonCustomOp")
class AddPythonCustomOp:
    def execute(self, x: Tensor, y: Tensor) -> None:
        ctx = get_execute_ctx()
        output = ctx.malloc_output_tensor(0, x.shape, x.format, x.data_type)
```

## 约束说明

- 仅可在当前`execute`调用栈内调用。
- 返回的`EagerOpExecutionContext`只能在当前`execute`内使用；由其返回的`Tensor`也只能在当前`execute`内使用。
