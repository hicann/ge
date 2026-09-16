# name

## 产品支持情况

全量芯片支持。

## 功能说明

获取ONNX节点的名称。

## 函数原型

```python
onnx_node.name -> str
```

## 参数说明

无

## 返回值说明

返回ONNX节点的名称，类型为`str`。

## 约束说明

- OnnxNode由GE构造并作为回调实参传入，用户不能自行构造。
- 属性为只读，仅支持读取，不支持修改。

## 调用示例

```python
node_name = node.name
```
