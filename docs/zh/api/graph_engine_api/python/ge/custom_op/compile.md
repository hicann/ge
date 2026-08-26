# compile

## 产品支持情况

全量芯片支持。

## 功能说明

Python自定义算子的图编译期回调。该回调依赖GE注册的Ascend IR算子原型；原生算子通过GE的`REG_OP`注册，Python算子原型由bridge同步注册。算子原型定义输入、输出和属性的名称、顺序、类型及输入输出类别。将实现类通过[register_op_impl](register_op_impl.md)注册并提供可调用的`compile`方法后，GE在图编译阶段调用该方法。回调可以读取schema参数，并通过[get_compile_ctx](get_compile_ctx.md)和[get_compile_platform_info](get_compile_platform_info.md)查询编译环境及平台信息。

`compile`回调只用于图编译，不在模型加载或模型执行阶段调用。

## 函数原型

```python
def compile(self, input_0, ..., output_0, ..., *, attr_0, ...) -> None
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

- `compile`只能以schema-bound形式使用。算子必须存在Ascend IR算子原型；否则在校验或调用时抛出`RuntimeError`。
- 回调不得声明可变位置参数或可变关键字参数。输入、输出和属性的数量、顺序或属性名称不匹配时，抛出`TypeError`。
- 回调返回值必须为`None`，并且必须声明`-> None`返回注解。
- `get_compile_ctx()`和`get_compile_platform_info()`只能在当前同步`compile`回调内调用。返回的上下文、平台信息、输入输出`Tensor`及Tensor属性视图均为借用对象，回调返回或抛出异常后失效。
- 编译上下文中的字符串、整数和`dict`等查询结果为Python值副本，可以在回调结束后继续使用。
- `compile`可以与同一实现类上的`execute`或`declare_launch_args`能力组合注册；各回调的上下文和生命周期相互独立。

## 调用示例

```python
from ge.custom_op import (
    get_compile_ctx,
    get_compile_platform_info,
    register_op_impl,
)
from ge.runtime import Tensor


@register_op_impl(op_type="AddCustom")
class AddCustom:
    def compile(self, x: Tensor, y: Tensor, z: Tensor, *, alpha: int) -> None:
        compile_ctx = get_compile_ctx()
        platform = get_compile_platform_info()
        option = compile_ctx.get_option("custom.compile.option")
        soc_version = platform.get_soc_version()
        core_num = platform.get_ai_core_num()
        # 根据输入、输出、属性及编译环境完成自定义编译逻辑。
        _ = (x, y, z, alpha, option, soc_version, core_num)
```
