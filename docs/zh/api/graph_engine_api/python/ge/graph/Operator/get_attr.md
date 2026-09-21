# get\_attr

## 产品支持情况

全量芯片支持。

## 功能说明

读取算子上指定名称的属性值。

## 函数原型

```python
Operator.get_attr(name: str) -> object
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| :--- | :--- | :--- |
| name | 输入 | 属性名称，类型为非空字符串。 |

## 返回值说明

返回属性值，具体类型取决于属性的存储类型，支持`int`、`float`、`bool`、`str`、`DataType`、`Tensor`及对应元素类型的列表。

## 约束说明

- 仅在回调执行期间可用，回调结束后调用会抛出`RuntimeError`。
- [`parse_operator`](../../onnx_plugin/OnnxPlugin/parse_operator.md)和[`decompose`](../../onnx_plugin/OnnxPlugin/decompose.md)回调中的source算子为只读对象，只能读取属性，不能写入。
- `name`不是非空字符串时，抛出`TypeError`。
- 属性不存在或读取失败时，抛出`RuntimeError`。

## 调用示例

```python
alpha = target.get_attr("alpha")
```
