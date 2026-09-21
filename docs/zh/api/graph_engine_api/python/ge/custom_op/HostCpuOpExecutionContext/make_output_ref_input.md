# make\_output\_ref\_input

## 产品支持情况

全量芯片支持。

## 功能说明

指定某输出的内存地址引用自某个输入。

## 函数原型

```python
make_output_ref_input(output_index: int, input_index: int) -> Tensor
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| :--- | :--- | :--- |
| output_index | 输入 | 输出索引。 |
| input_index | 输入 | 输入索引。 |

## 返回值说明

| 类型 | 说明                           |
| :--- |:-------------------------------|
| Tensor | output_index对应的输出张量。 |

## 约束说明

- 仅可在当前`execute`调用栈内调用。
- 返回的`Tensor`只能在当前`execute`内使用。
- 两个索引必须位于当前节点的输入、输出实例范围内，查找或设置失败时抛出`RuntimeError`。
- 本接口中output_index对应的输出和input_index对应的输入，必须在register_op中通过[mutates_args](../register_op.md#参数说明)声明输出引用输入的关系。未通过mutates_args声明时，调用此接口会失败。

## 调用示例

以下示例通过register_op定义一个简单的算子原型。mutates_args=("x",)表示第0个输出引用输入x。

```python
from ge.custom_op import register_op
from ge.runtime import TensorDesc


@register_op(op_type="IdentityCustom", mutates_args=("x",))
def infer_meta(x: TensorDesc) -> TensorDesc:
    return x
```

在execute实现中，使用make_output_ref_input(0, 0)将第0个输出设置为引用第0个输入的内存地址。

```python
from ge.custom_op import get_execute_ctx


def execute(self, x) -> None:
    ctx = get_execute_ctx()
    output = ctx.make_output_ref_input(0, 0)
```
