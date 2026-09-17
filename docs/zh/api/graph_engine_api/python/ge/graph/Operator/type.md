# type

## 产品支持情况

全量芯片支持。

## 功能说明

获取算子类型。ONNX Plugin回调中，该类型为descriptor中`target`参数指定的GE目标算子类型。

## 函数原型

```python
operator.type -> str
```

## 参数说明

无

## 返回值说明

返回算子类型，类型为`str`。

## 约束说明

- 仅在回调执行期间可用，回调结束后调用会抛出`RuntimeError`。
- Operator对象由GE构造并作为回调实参传入，用户不能自行构造。

## 调用示例

```python
op_type = target.type
```
