# 简介

OnnxNode类是ONNX节点的只读值对象，由GE在解析阶段根据ONNX模型的`NodeProto`构造，并作为[`parse_node`](../OnnxPlugin/parse_node.md)回调的实参传入（另一个实参为[`Operator`](../../graph/Operator/overview.md)对象）。对象由GE创建和销毁，用户不需要也无法自行构造。

OnnxNode仅提供以下只读属性，不提供任何修改方法：

- [`name`](name.md)：ONNX节点名称。
- [`origin_type`](origin_type.md)：节点的完整origin type。
- [`inputs`](inputs.md)：输入tensor名称元组。
- [`outputs`](outputs.md)：输出tensor名称元组。
- [`attrs`](attrs.md)：节点属性的只读视图。

访问示例（`parse_node`回调内）：

```python
from ge.graph import Operator
from ge.onnx_plugin import OnnxNode, onnx_plugin


elu = onnx_plugin(
    source="Elu",
    domain="ai.onnx",
    opsets=range(8, 19),
    target="Elu",
)


@elu.parse_node
def parse_elu(node: OnnxNode, target: Operator) -> None:
    alpha = node.attrs.get("alpha", 1.0)
    target.set_attr("alpha", alpha)
```
