# declare\_launch\_args

## 产品支持情况

全量芯片支持。

## 功能说明

Python自定义算子的声明式kernel启动参数回调。将实现类通过[register_op_impl](register_op_impl.md)注册并提供可调用的`declare_launch_args`方法后，GE根据Ascend IR算子原型组装输入、输出和属性参数，并在编译阶段调用该方法。回调通过[get_declare_launch_args_ctx](get_declare_launch_args_ctx.md)创建`AnnotatedKernelArgs`、申请workspace，并提交`AnnotatedKernelLaunchInfo`。

## 函数原型

```python
def declare_launch_args(self, input_0, ..., output_0, ..., *, attr_0, ...) -> None
```

上面的参数名仅表示参数位置。实际参数数量和属性名称由算子的Ascend IR算子原型决定。

## 参数说明

GE按Ascend IR算子原型绑定参数，顺序如下：

| 参数 | 绑定规则 |
| :--- | :--- |
| 输入参数 | 位于参数列表前部。required input传入`Tensor`，optional input传入`Optional[Tensor]`，dynamic input传入`List[Tensor]`。 |
| 输出参数 | 位于所有输入参数之后。required output传入`Tensor`，dynamic output传入`List[Tensor]`。 |
| 属性参数 | 位于所有输入、输出参数之后，必须使用keyword-only参数；参数名称、顺序和类型与Ascend IR算子原型一致。 |

参数类型由Ascend IR算子原型决定。参数提供类型注解时，注解必须与对应的输入、输出或属性类型一致。

## 约束说明

- `declare_launch_args`只能以schema-bound形式使用。算子必须存在Ascend IR算子原型；否则在校验或调用时抛出`RuntimeError`。
- 回调不得声明可变位置参数或可变关键字参数。输入、输出和属性的数量、顺序或属性名称不匹配时，抛出`TypeError`。
- 回调返回值必须为`None`，并且必须声明`-> None`返回注解。
- `get_declare_launch_args_ctx()`只能在当前同步`declare_launch_args`回调内调用。返回的`AnnotatedArgsContext`、`AnnotatedKernelArgs`、`WorkspaceAddr`及其派生的借用对象只能在当前回调内使用，回调返回或抛出异常后失效。
- `AnnotatedKernelArgs.append_input`和`append_output`的`instance_index`使用当前计算节点输入、输出的实例平铺索引；动态输入或输出展开后的实例使用连续索引。
- 调用`AnnotatedArgsContext.add_launch`后，传入的`AnnotatedKernelArgs`会被消费，不能再次使用。

## 调用示例

```python
from ge.custom_op import (
    AnnotatedKernelLaunchInfo,
    get_declare_launch_args_ctx,
    register_op_impl,
)
from ge.runtime import Tensor


kernel_bin = b"..."


@register_op_impl(op_type="AnnotatedAddCustom")
class AnnotatedAddCustom:
    def declare_launch_args(self, x1: Tensor, x2: Tensor, y: Tensor) -> None:
        ctx = get_declare_launch_args_ctx()
        args = ctx.create_kernel_args()
        args.append_input(0, x1)
        args.append_input(1, x2)
        args.append_output(0, y)
        ctx.add_launch(
            AnnotatedKernelLaunchInfo(
                kernel_name="add_custom",
                kernel_bin=kernel_bin,
                block_dim=8,
                stream_id=ctx.get_stream_id(),
            ),
            args,
        )
```
