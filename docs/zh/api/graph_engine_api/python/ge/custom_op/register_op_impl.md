# register\_op\_impl

## 产品支持情况

全量芯片支持。

## 功能说明

注册Python自定义算子实现类。支持实现`execute`、`compile`和`declare_launch_args`方法，这些方法可以组合使用。

## 函数原型

```python
register_op_impl(*, op_type: str) -> callable
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| :--- | :--- | :--- |
| op_type | 输入 | 自定义算子类型。必须是非空字符串，且在实现注册表中唯一。 |

## 约束说明

- 注册的实现类必须是具体类，并且至少实现一个受支持的方法。具体回调约束参见[execute](execute.md)、[compile](compile.md)和[declare_launch_args](declare_launch_args.md)。
- `op_type`不合法、注册的实现类不是具体类，或实现类未提供受支持的方法时，抛出`TypeError`。`op_type`重复注册发生冲突时，抛出`ValueError`。

## 调用示例

```python
from ge.custom_op import get_execute_ctx, register_op_impl
from ge.runtime import Tensor


@register_op_impl(op_type="AddCustom")
class AddCustom:
    def execute(self, x: Tensor, y: Tensor) -> None:
        ctx = get_execute_ctx()
        output = ctx.malloc_output_tensor(0, x.shape, x.format, x.data_type)
        # 使用x、y、output和ctx.get_stream()发起当前算子的执行。
```
