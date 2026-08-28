# execute

## 产品支持情况

全量芯片支持。

## 功能说明

Python自定义算子的运行时执行回调。将实现类通过[register_op_impl](register_op_impl.md)注册并提供可调用的`execute`方法后，GE根据算子原型组装输入和属性参数，并在算子执行阶段调用该方法。

`execute`回调不接收输出参数，也不通过返回值传递输出。可通过[get_execute_ctx](get_execute_ctx.md)获取`EagerOpExecutionContext`，申请输出内存或建立输出与输入之间的Ref关系。

## 函数原型

`execute`方法的签名遵循以下模式：

```python
def execute(self, input_0, ..., *, attr_0, ...) -> None
```

上面的参数名仅表示参数位置。实际参数数量和属性名称由算子的原型决定。

## 参数说明

| 参数 | 绑定规则                             |
| :--- |:-------------------------------------|
| 输入参数 | 位于参数列表前部。required input传入`Tensor`，optional input传入`Optional[Tensor]`，dynamic input传入`List[Tensor]`。 |
| 属性参数 | 位于所有输入参数之后，必须使用keyword-only参数；参数名称、顺序和类型与算子原型一致。 |

参数类型由算子原型决定。参数提供类型注解时，注解必须与对应的输入或属性类型一致。

## 约束说明

- `execute`只能以schema-bound形式使用。算子必须存在算子原型；否则在校验或调用时抛出`RuntimeError`。
- 回调不得声明可变位置参数或可变关键字参数。输入和属性的数量、顺序或属性名称不匹配时，抛出`TypeError`。
- `execute`无需为输入参数指定类型提示；但是，任何指定的类型提示都将根据算子原型进行验证，以确保一致性。回调必须显式声明`-> None`返回注解，并且返回值必须为`None`。
- 回调返回值必须为`None`，并且必须声明`-> None`返回注解。
- `get_execute_ctx()`只能在当前同步`execute`回调内调用。返回的`EagerOpExecutionContext`、`Tensor`和`RuntimeAttrs`等借用对象只能在当前回调内使用，回调返回或抛出异常后失效。

## 调用示例

```python
from ge.custom_op import get_execute_ctx, register_op_impl
from ge.runtime import Tensor


@register_op_impl(op_type="AddPythonCustomOp")
class AddPythonCustomOp:
    def execute(self, x: Tensor, y: Tensor) -> None:
        ctx = get_execute_ctx()
        output = ctx.malloc_output_tensor(0, x.shape, x.format, x.data_type)
        # 使用x、y、output和ctx.get_stream()发起当前算子的执行。
```
