# 简介

`TensorDesc`类用于表示Python自定义算子`infer_meta`的张量元信息，保存Tensor的逻辑Shape、StorageShape和DataType。该对象由Python代码持有，可以作为`infer_meta`的返回值。

注意：该类型与[`ge.graph.TensorDesc`](../../graph/TensorDesc/overview.md)不同。`ge.graph.TensorDesc`用于图构建侧的Tensor描述，包含Format、OriginShape以及对应的`get_*`/`set_*`接口；`ge.runtime.TensorDesc`仅用于Python原型的元信息输入和输出。

以下示例创建两个输入TensorDesc，并调用`infer_meta`返回输出TensorDesc。

```python
from ge.custom_op import register_op
from ge.graph import DataType
from ge.runtime import TensorDesc


@register_op(op_type="AddCustom")
def infer_meta(x: TensorDesc, y: TensorDesc) -> TensorDesc:
    return TensorDesc(x.shape, x.data_type)


x = TensorDesc([2, 3], DataType.DT_FLOAT)
y = TensorDesc([2, 3], DataType.DT_FLOAT)
z = infer_meta(x, y)
```
