# register\_kernel

## 产品支持情况

全量芯片支持。

## 功能说明

为register_op_impl实现类中的execute声明执行backend。同类中定义多个同名execute时，装饰器会在后一个定义覆盖前保存函数对象。

## 函数原型

```python
register_kernel(*, backend: OpBackend) -> callable
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| :--- | :--- | :--- |
| backend | 输入 | [OpBackend](OpBackend.md)枚举成员。 |

## 约束说明

- 只能装饰@register_op_impl修饰的实现类中的execute，支持实例方法、staticmethod和classmethod；与后两者组合时，register_kernel必须位于最外层。
- 一个实现类对同一backend最多声明一个execute；为多个backend分别声明时，每个execute都必须使用register_kernel，不得与未装饰实现混用。

## 调用示例

```python
from ge.custom_op import OpBackend, register_kernel, register_op_impl


@register_op_impl(op_type="MyCustomOp")
class MyCustomOpImpl:
    @register_kernel(backend=OpBackend.DEVICE)
    def execute(self, x, *, alpha: float) -> None:
        ...

    @register_kernel(backend=OpBackend.HOST)
    def execute(self, x, *, alpha: float) -> None:
        ...
```
