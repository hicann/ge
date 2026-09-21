# get\_output\_tensor

## 产品支持情况

全量芯片支持。

## 功能说明

获取index指定的输出张量。

## 函数原型

```python
get_output_tensor(index: int) -> Tensor
```

## 参数说明

| 参数名 | 输入/输出 | 描述                  |
| :--- | :--- |:----------------------|
| index | 输入 | 输出索引。 |

## 返回值说明

| 类型 | 说明         |
| :--- |:-------------|
| Tensor | 输出张量。 |

## 约束说明

- 仅可在当前`execute`调用栈内调用。
- 返回的`Tensor`只能在当前`execute`内使用。
- `index`必须位于当前计算节点输出实例的范围内，输出不可用时抛出`RuntimeError`。

## 调用示例

```python
from ge.custom_op import get_execute_ctx


def execute(self, x) -> None:
    ctx = get_execute_ctx()
    output = ctx.get_output_tensor(0)
```
