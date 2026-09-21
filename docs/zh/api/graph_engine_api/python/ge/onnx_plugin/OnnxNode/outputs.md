# outputs

## 产品支持情况

全量芯片支持。

## 功能说明

获取ONNX节点的全部输出tensor名称。

## 函数原型

```python
onnx_node.outputs -> tuple
```

## 参数说明

无

## 返回值说明

返回输出tensor名称元组，类型为`tuple[str, ...]`，元素顺序与ONNX节点定义一致。

## 约束说明

- 返回的元组为只读快照，修改该元组不影响ONNX节点。
- OnnxNode由GE构造并作为回调实参传入，用户不能自行构造。

## 调用示例

```python
output_names = node.outputs
```
