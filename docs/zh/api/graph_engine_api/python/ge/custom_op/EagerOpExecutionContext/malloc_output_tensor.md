# malloc\_output\_tensor

## 产品支持情况

全量芯片支持。

## 功能说明

为某个输出Tensor申请Device内存，同时初始化输出Tensor的基本信息。
该输出Tensor的内存由Context构造方管理。接口调用者不需要主动释放。

## 函数原型

```python
malloc_output_tensor(index: int, shape: StorageShape, format: StorageFormat, dtype: DataType) -> Tensor
```

## 参数说明

| 参数名 | 输入/输出 | 描述                  |
| :--- | :--- |:----------------------|
| index | 输入 | 输出索引。            |
| shape | 输入 | 输出张量的shape。     |
| format | 输入 | 输出张量的format。    |
| dtype | 输入 | 输出张量的data type。 |

## 返回值说明

| 类型 | 说明                                   |
| :--- |:---------------------------------------|
| Tensor | 输出张量。 |

## 约束说明

- 仅可在当前`execute`调用栈内调用。
- 返回的`Tensor`只能在当前`execute`内使用。
- `shape`和`format`必须是有效的存储描述对象，`index`必须位于当前节点的输出实例范围内，申请或初始化失败时抛出`RuntimeError`。

## 调用示例

```python
from ge.custom_op import get_execute_ctx


def execute(self, x) -> None:
    ctx = get_execute_ctx()
    output = ctx.malloc_output_tensor(0, x.shape, x.format, x.data_type)
```
