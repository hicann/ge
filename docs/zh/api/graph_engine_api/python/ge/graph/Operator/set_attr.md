# set\_attr

## 产品支持情况

全量芯片支持。

## 功能说明

设置算子上指定名称的属性值。

## 函数原型

```python
Operator.set_attr(name: str, value: object) -> None
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| :--- | :--- | :--- |
| name | 输入 | 属性名称，类型为非空字符串。 |
| value | 输入 | 属性值。支持`int`、`float`、`bool`、`str`、`DataType`、`Tensor`，以及元素类型一致的`list`（元素为数值、`bool`、`str`或`DataType`）。列表必须为`list`类型且非空，数值列表的元素可为`int`或`float`。 |

## 返回值说明

无。

## 约束说明

- 仅在回调执行期间可用，回调结束后调用会抛出`RuntimeError`。
- [`parse_operator`](../../onnx_plugin/OnnxPlugin/parse_operator.md)和[`decompose`](../../onnx_plugin/OnnxPlugin/decompose.md)回调中的source算子为只读对象，调用会抛出`RuntimeError`。
- `name`不是非空字符串时，抛出`TypeError`。
- `value`为不支持的类型（包括空列表、元素类型不一致的列表等）时，抛出`ValueError`。
- 本接口不做属性名校验，属性名未在算子原型中声明时不报错，问题可能在后续图校验阶段暴露。

## 调用示例

```python
target.set_attr("alpha", 1.0)
```
