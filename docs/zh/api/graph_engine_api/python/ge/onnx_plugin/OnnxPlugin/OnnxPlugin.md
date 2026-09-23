# OnnxPlugin

## 产品支持情况

全量芯片支持。

## 功能说明

OnnxPlugin类是ONNX Plugin的descriptor（描述符），由[`onnx_plugin()`](../onnx_plugin.md)创建，用于声明ONNX原始算子到GE目标算子的映射，并通过[`parse_node`](parse_node.md)、[`parse_operator`](parse_operator.md)、[`decompose`](decompose.md)方法绑定解析回调。

类的详细说明、回调选择建议与使用示例参见[简介](overview.md)。
