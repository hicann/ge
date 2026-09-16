# parse\_operator

## 产品支持情况

全量芯片支持。

## 功能说明

为descriptor绑定基于Operator的参数解析回调。解析阶段，解析器根据ONNX节点自动创建source算子和target算子，并将两者传入回调；回调从source算子读取属性等信息，补充target算子。

## 函数原型

```python
OnnxPlugin.parse_operator(fn: Callable[..., None]) -> Callable[..., None]
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| :--- | :--- | :--- |
| fn | 输入 | 被绑定的参数解析回调。类型为Python函数，回调签名为`(source: Operator, target: Operator) -> None`。`source`为解析器根据ONNX节点自动创建的只读算子，`target`为目标算子。 |

## 返回值说明

返回原回调函数，装饰过程不改变原函数的调用方式。

## 约束说明

- `source`为只读对象，对其调用`set_attr`或端口注册方法会抛出`RuntimeError`；`get_attr`、`name`和`type`可正常使用。
- source算子的属性读取约定：解析器将ONNX节点的属性列表整体序列化为JSON字符串，存入名为`attribute`的属性，回调需要通过`json.loads(source.get_attr("attribute"))`解析后，按`name`字段查找属性值。
- 同一descriptor同时绑定了[`parse_node`](parse_node.md)时，解析器优先调用本回调。
- 回调返回值、生命周期和异常约束与[`parse_node`](parse_node.md)一致。
- `fn`不是Python函数（例如类的实例、内置函数等其他可调用对象）时，抛出`TypeError`。
- 同一descriptor重复绑定`parse_operator`回调时，抛出`ValueError`。

## 调用示例

以下示例将source算子上的JSON属性整体搬运到target算子，无需预知属性名即可完成属性传递：

```python
from ge.graph import Operator
from ge.onnx_plugin import onnx_plugin


plugin = onnx_plugin(
    source="GroupNormRelu",
    domain="example.domain",
    opsets=(1,),
    target="GroupNormRelu",
)


@plugin.parse_operator
def parse_group_norm(source: Operator, target: Operator) -> None:
    target.set_attr("attribute", source.get_attr("attribute"))
```

`attribute`属性的JSON结构示意如下。每个条目的`name`为属性名，`type`为属性类型的枚举值（ONNX `AttributeProto`的取值，如`1`表示FLOAT），取值字段按属性实际内容输出：

```json
{
  "attribute": [
    {"name": "alpha", "type": 1, "f": "1"}
  ]
}
```

各取值字段在JSON中的形态与属性类型对应，按条目中实际存在的字段取用即可：

- `f`（FLOAT类型）：字符串（浮点数经`%g`格式化，约6位有效数字），使用时需转换为数值，如`float(item["f"])`。
- `i`（INT类型）：数字。
- `s`（STRING类型）：字符串（内容经转义处理，普通文本可直接使用）。
- `t`（TENSOR类型）、`g`（GRAPH类型）：嵌套对象，结构与ONNX `TensorProto`/`GraphProto`的字段对应。
- `floats`、`ints`（列表类型）：数字数组；`strings`：字符串数组；`tensors`、`graphs`：对象数组。注意`floats`的元素为原生数值，与标量`f`的字符串形态不同；`strings`的元素不经转义处理，与标量`s`不同。

复合类型属性（`t`、`g`及其列表）以嵌套对象形式提供，可读取和检查；如需将其转换为`Tensor`等GE对象写入目标算子，需要按上述字段结构自行解码后构造，`set_attr`不支持直接传入`dict`。

节点无属性时，source算子上不存在名为`attribute`的属性，`get_attr("attribute")`会抛出`RuntimeError`。

需要按名读取单个属性时，解析后按`name`字段查找。以下为回调内片段，读取FLOAT类型的`alpha`属性：

```python
import json

attrs = json.loads(source.get_attr("attribute"))["attribute"]
alpha = None
for item in attrs:
    if item["name"] == "alpha":
        alpha = float(item["f"])
        break
```
