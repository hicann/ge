# register\_op\_impl

## 产品支持情况

全量芯片支持。

## 头文件/库文件

- 头文件：无
- 库文件：ge_custom_op_native.so、libge_python_custom_op_bridge.so

## 功能说明

注册Python自定义算子实现类。实现类中的可调用`declare_launch_args`方法会注册为`annotated_args`能力。GE在静态图编译阶段调用该能力，由实现类声明kernel launch参数。

## 函数原型

```python
register_op_impl(*, op_type: str) -> callable
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| :--- | :--- | :--- |
| op_type | 输入 | 自定义算子类型。必须是非空字符串，且在实现注册表中唯一。 |

## 返回值说明

| 类型 | 说明 |
| :--- | :--- |
| callable | 返回类装饰器。装饰器注册实现类后返回该类，并设置`__ge_op_impl_descriptor__`属性。 |

## 约束说明

- 被装饰对象必须是具体类，并且至少实现一个受支持的可调用能力方法。当前受支持的方法为`execute`和`declare_launch_args`；后者映射为`annotated_args`能力。
- `op_type`不合法、被装饰对象不是具体类，或实现类未提供受支持的可调用能力方法时，抛出`TypeError`。`op_type`重复注册发生冲突时，抛出`ValueError`。
- 注册阶段只收集实现类的能力，不校验`declare_launch_args`的业务参数签名。在可获得canonical IR后的实现描述符校验阶段，`declare_launch_args`的参数按IR中输入、输出、仅限关键字属性的顺序绑定。
- `declare_launch_args`的必选输入和必选输出参数类型为`Tensor`，可选输入参数类型为`Optional[Tensor]`，动态输入和动态输出参数类型为`List[Tensor]`。属性参数必须为仅限关键字参数，并与canonical IR中的属性名称和类型一致。
- `declare_launch_args`的返回注解和返回值均必须为`None`。签名或返回值不符合要求时，抛出`TypeError`。
