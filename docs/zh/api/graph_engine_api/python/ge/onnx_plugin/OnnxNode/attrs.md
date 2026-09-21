# attrs

## 产品支持情况

全量芯片支持。

## 功能说明

获取ONNX节点全部属性的只读视图。

## 函数原型

```python
onnx_node.attrs -> Mapping[str, object]
```

## 参数说明

无

## 返回值说明

返回以属性名为键的字典只读视图，属性值按ONNX属性类型转换为Python对象：

| ONNX属性类型 | Python类型 |
| :--- | :--- |
| FLOAT | `float` |
| INT | `int` |
| STRING | `str` |
| FLOATS | `list[float]` |
| INTS | `list[int]` |
| STRINGS | `list[str]` |

## 约束说明

- 返回视图为只读视图，修改会抛出`TypeError`。
- 属性名必须非空，存在空属性名时当前节点解析失败。
- 仅支持上表所列的标量及同类型列表属性，TENSOR、GRAPH、SPARSE_TENSOR等复合类型属性不支持，解析到此类属性时当前节点解析失败。
- `attrs`是整体视图，没有按单个属性名读取的接口。节点存在上述不支持的属性时，访问`attrs`整体失败，该节点上的其他普通属性也无法读取；此类节点需改用[`parse_operator`](../OnnxPlugin/parse_operator.md)回调，通过source算子读取JSON格式的属性。
- OnnxNode由GE构造并作为回调实参传入，用户不能自行构造。

## 调用示例

```python
alpha = node.attrs.get("alpha", 1.0)
```
