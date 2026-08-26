# register\_op\_impl

## 产品支持情况

全量芯片支持。

## 功能说明

注册Python自定义算子实现类。装饰器反射类上的可调用方法：`execute`、`compile`、`declare_launch_args`分别声明`eager_execute`、`compilable`、`annotated_args`能力。三种能力可以组合使用，具体回调约束参见[compile](compile.md)和[declare_launch_args](declare_launch_args.md)。

## 函数原型

```python
register_op_impl(*, op_type: str) -> callable
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| :--- | :--- | :--- |
| op_type | 输入 | 自定义算子类型。必须是非空字符串，且在实现注册表中唯一。 |

## 约束说明

- 被装饰对象必须是具体类，并且至少实现一个受支持的可调用能力方法。当前受支持的方法为`execute`、`compile`和`declare_launch_args`。
- `op_type`不合法、被装饰对象不是具体类，或实现类未提供受支持的可调用能力方法时，抛出`TypeError`。`op_type`重复注册发生冲突时，抛出`ValueError`。
- 注册阶段只收集实现类的能力，不校验`declare_launch_args`的业务参数签名。在可获得Ascend IR算子原型后的实现描述符校验阶段，`declare_launch_args`的参数按Ascend IR算子原型中输入、输出、仅限关键字属性的顺序绑定。
- `compile`只支持schema-bound形式：参数按Ascend IR算子原型的输入、输出顺序绑定，属性使用名称、顺序与Ascend IR算子原型一致的keyword-only参数，返回注解和返回值均必须为`None`。它在图编译阶段调用；回调中通过[get_compile_ctx](get_compile_ctx.md)查询编译环境。
- `declare_launch_args`的必选输入和必选输出参数类型为`Tensor`，可选输入参数类型为`Optional[Tensor]`，动态输入和动态输出参数类型为`List[Tensor]`。属性参数必须为仅限关键字参数，并与Ascend IR算子原型中的属性名称和类型一致。
- `declare_launch_args`的返回注解和返回值均必须为`None`。签名或返回值不符合要求时，抛出`TypeError`。

## 调用示例

```python
from ge.custom_op import register_op_impl


@register_op_impl(op_type="AddCustom")
class AddCustom:
    def execute(self, x, y):
        return x + y
```
