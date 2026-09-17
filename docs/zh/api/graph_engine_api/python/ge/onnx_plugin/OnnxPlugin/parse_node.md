# parse\_node

## 产品支持情况

全量芯片支持。

## 功能说明

为descriptor绑定一对一参数解析回调。解析阶段，解析器按origin type匹配到该回调后，将ONNX节点和已创建的目标算子传入回调；回调读取ONNX节点信息，补充目标算子的属性和端口。

## 函数原型

```python
OnnxPlugin.parse_node(fn: Callable[..., None]) -> Callable[..., None]
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| :--- | :--- | :--- |
| fn | 输入 | 被绑定的一对一参数解析回调。类型为Python函数，回调签名为`(node: OnnxNode, target: Operator) -> None`。`node`为ONNX节点只读值对象，`target`为解析器创建的目标算子，回调通过`set_attr`和`register_*`系列方法补充其属性和端口。 |

## 返回值说明

返回原回调函数，装饰过程不改变原函数的调用方式。

## 约束说明

- 回调返回值必须为`None`，返回其他值时当前节点解析失败。
- 回调内抛出的异常会导致当前节点解析失败，目标算子不会加入图。
- `target`仅在回调执行期间有效，回调返回或抛出异常后继续访问会抛出`RuntimeError`。
- 通过`node.attrs`读取属性时仅支持标量及同类型列表（`int`、`float`、`str`及对应列表）；`TENSOR`、`GRAPH`等复合类型属性不支持，节点存在此类属性时`attrs`整体读取失败。类型对照与完整约束参见[`attrs`](../OnnxNode/attrs.md)；此类节点需改用[`parse_operator`](parse_operator.md)。
- 目标算子为动态输入输出原型（如`PartitionedCall`）时，必须在回调中调用`register_input`、`register_output`等方法注册端口，否则解析器连线阶段失败；目标算子为静态原型（如`Elu`）时不需要注册端口。
- 同一descriptor同时绑定了[`parse_operator`](parse_operator.md)时，解析器优先调用`parse_operator`，本回调不会被调用。
- `fn`不是Python函数（例如类的实例、内置函数等其他可调用对象）时，抛出`TypeError`。
- 同一descriptor重复绑定`parse_node`回调时，抛出`ValueError`。

## 调用示例

以下示例读取ONNX节点的`alpha`属性并写入目标算子：

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
    if not isinstance(alpha, (int, float)):
        raise TypeError("Elu alpha must be a number")
    target.set_attr("alpha", alpha)
```

以下示例按ONNX节点的输入数量为动态输入目标算子注册端口：

```python
from ge.graph import Operator
from ge.onnx_plugin import OnnxNode, onnx_plugin


sum_plugin = onnx_plugin(
    source="Sum",
    domain="ai.onnx",
    opsets=range(1, 23),
    target="Sum",
)


@sum_plugin.parse_node
def parse_sum(node: OnnxNode, target: Operator) -> None:
    count = len(node.inputs)
    if count == 0:
        raise ValueError("Sum requires at least one input")
    target.register_dynamic_input("x", count)
    target.set_attr("N", count)
```
