# 简介

OnnxPlugin类是ONNX Plugin的descriptor（描述符）。一个descriptor实例声明一种ONNX原始算子到GE目标算子的映射，并持有绑定的解析回调，由[`onnx_plugin()`](../onnx_plugin.md)创建。

一个descriptor支持绑定以下回调：

- [`parse_node`](parse_node.md)：一对一参数解析回调，对应C++的[`ParseParamsFn`](../../../../../../user_guides/custom_op/custom_op_v2/custom_op_development_guide.md#64-onnx-入图)。
- [`parse_operator`](parse_operator.md)：基于Operator的参数解析回调，对应C++的[`ParseParamsByOperatorFn`](../../../../../../user_guides/custom_op/custom_op_v2/custom_op_development_guide.md#64-onnx-入图)。
- [`decompose`](decompose.md)：一对多分解回调，对应C++的[`ParseOpToGraphFn`](../../../../../../user_guides/custom_op/custom_op_v2/custom_op_development_guide.md#64-onnx-入图)。

三类回调对应解析流程的不同阶段，可按需组合：

- `parse_node`与`parse_operator`属于同一参数解析阶段，两者同时绑定时仅`parse_operator`生效，`parse_node`不会被调用。
- `decompose`独立于参数解析阶段，可与任一参数解析回调组合，也可单独绑定。

回调通过装饰器方式绑定，装饰器返回原函数，绑定回调时完成descriptor注册。同一类回调重复绑定时抛出`ValueError`。descriptor创建后，`source`、`domain`、`opsets`和`target`不可修改。

回调选择建议：

- 只需参数解析时优先用`parse_node`：按属性名直接读取ONNX节点属性，最直观。
- 节点属性含张量、子图等复合类型时只能用`parse_operator`：[`OnnxNode.attrs`](../OnnxNode/attrs.md)不支持复合类型属性，遇到此类属性的节点读取会整体失败。
- 目标算子在GE中无一对应的实现、需用已有算子组合表达时，绑定`decompose`，在回调中构建替代子图。

使用示例：

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
    target.set_attr("alpha", node.attrs.get("alpha", 1.0))
```
