# origin\_type

## 产品支持情况

全量芯片支持。

## 功能说明

获取ONNX节点的完整origin type。

## 函数原型

```python
onnx_node.origin_type -> str
```

## 参数说明

无

## 返回值说明

返回节点的完整origin type，类型为`str`，格式为`domain::opset::算子类型`，例如`ai.onnx::11::Elu`。其中`domain`取ONNX模型文件中节点的domain字段，该字段允许为空，ONNX标准规定为空时即标准域`ai.onnx`，因此显示为`ai.onnx`；`opset`为该domain在模型中的opset版本。注意：插件注册参数[`domain`](../onnx_plugin.md)要求显式写非空字符串，与本节点的domain字段是两回事。

## 约束说明

- OnnxNode由GE构造并作为回调实参传入，用户不能自行构造。
- 属性为只读，仅支持读取，不支持修改。

## 调用示例

```python
origin_type = node.origin_type
```
