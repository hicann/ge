# decompose

## 产品支持情况

全量芯片支持。

## 功能说明

为descriptor绑定一对多分解回调。解析阶段，解析器将source算子传入回调；回调使用ES构图接口构建替代子图并返回，解析器将该子图展开入图，替代单一目标算子。

## 函数原型

```python
OnnxPlugin.decompose(fn: Callable[..., object]) -> Callable[..., object]
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| :--- | :--- | :--- |
| fn | 输入 | 被绑定的一对多分解回调。类型为Python函数，回调签名为`(source: Operator) -> Graph`。`source`为参数解析阶段产出的算子的只读视图，返回值为替代子图，类型为`ge.graph.Graph`。 |

## 返回值说明

返回原回调函数，装饰过程不改变原函数的调用方式。

## 约束说明

- 回调返回值必须为`ge.graph.Graph`对象，返回其他值（含`None`）时当前节点解析失败。
- `source`为只读对象，对其调用`set_attr`或端口注册方法会抛出`RuntimeError`。
- `source`的属性和端口来自参数解析阶段的产出（[`parse_node`](parse_node.md)或[`parse_operator`](parse_operator.md)回调）：`get_attr`读取到的属性、已注册的端口均由该回调写入。目标算子为动态输入输出原型时，参数解析回调中必须完成端口注册。
- 分解场景下目标算子仅作为占位，解析完成后被替代子图替换；配合参数解析回调使用时，`target`通常选用`PartitionedCall`——GE中承载子图的通用算子，自身不做计算、端口定义灵活，便于在参数解析回调中按分解需要注册端口。
- 只绑定`decompose`（不绑定任何参数解析回调）时，参数解析阶段仅写入解析器内部属性、不注册端口：`source`上没有可读取的业务属性，且目标算子不能是动态输入输出原型（如`PartitionedCall`），否则解析器连线阶段失败，应选用端口已在算子原型中静态定义的`target`。
- 构图使用[`GraphBuilder`](../../es/GraphBuilder/GraphBuilder.md)等ES构图接口，`GraphBuilder`在回调内部创建；`ge.es.math`、`ge.es.nn`等构图算子来自外部ES插件包，使用前需确保已安装。
- `fn`不是Python函数（例如类的实例、内置函数等其他可调用对象）时，抛出`TypeError`。
- 同一descriptor重复绑定`decompose`回调时，抛出`ValueError`。

## 调用示例

以下示例将ThresholdedRelu节点分解为`Threshold`与`Mul`两个算子（完整样例参见[ONNX Plugin样例](../../../../../../../../examples/onnx_plugin/README.md)）：

```python
from ge.es import GraphBuilder
from ge.es.math import Mul
from ge.es.nn import Threshold
from ge.graph import Graph, Operator
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


@thresholded_relu.decompose
def decompose_thresholded_relu(source: Operator) -> Graph:
    alpha = float(source.get_attr("alpha"))
    builder = GraphBuilder("thresholded_relu_decomposition")
    x = builder.create_input(0)
    mask = Threshold(x, threshold=alpha)
    output = Mul(x, mask)
    return builder.build_and_reset([output])
```
