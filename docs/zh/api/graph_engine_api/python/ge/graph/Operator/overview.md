# 简介

Operator类是回调期间对GE算子的受控包装，在ONNX Plugin回调（[`parse_node`](../../onnx_plugin/OnnxPlugin/parse_node.md)、[`parse_operator`](../../onnx_plugin/OnnxPlugin/parse_operator.md)、[`decompose`](../../onnx_plugin/OnnxPlugin/decompose.md)）中由GE作为实参传入，用于读写算子的属性及输入输出等定义信息。

对象约束：

- Operator对象由GE构造并传入回调，用户不能直接构造。
- 对象仅在回调执行期间有效，回调返回或抛出异常后失效，继续调用其方法会抛出`RuntimeError`。
- `parse_operator`和`decompose`回调中的source算子为只读对象，`set_attr`和端口注册方法会抛出`RuntimeError`。
- 对象不支持copy和pickle。
- `ge.graph.Operator`与`ge.graph.Node`是不同对象：`Operator`包装解析回调期间的算子；`Node`包装已加入图的节点，两者不共享底层对象。

成员如下：

- [`name`](name.md)：算子名称。
- [`type`](type.md)：算子类型。
- [`get_attr`](get_attr.md)：读取算子属性。
- [`set_attr`](set_attr.md)：设置算子属性。
- [`register_input`](register_input.md)：注册必选输入端口。
- [`register_optional_input`](register_optional_input.md)：注册可选输入端口。
- [`register_output`](register_output.md)：注册输出端口。
- [`register_dynamic_input`](register_dynamic_input.md)：注册动态输入端口组。
- [`register_dynamic_output`](register_dynamic_output.md)：注册动态输出端口组。

使用示例（`parse_node`回调内为动态输入输出目标算子补充属性与端口；`target`选用动态输入输出原型`PartitionedCall`，其端口需在回调中注册）：

```python
from ge.graph import Operator
from ge.onnx_plugin import OnnxNode, onnx_plugin


thresholded_relu = onnx_plugin(
    source="ThresholdedRelu",
    domain="example.domain",
    opsets=(1,),
    target="PartitionedCall",
)


@thresholded_relu.parse_node
def parse_thresholded_relu(node: OnnxNode, target: Operator) -> None:
    target.set_attr("alpha", node.attrs.get("alpha", 1.0))
    target.register_input("x")
    target.register_output("y")
```
