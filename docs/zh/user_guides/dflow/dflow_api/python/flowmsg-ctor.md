# FlowMsg构造函数

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

FlowMsg的构造函数。

## 函数原型

```python
__init__(self, flow_msg: flowfunc_wrapper.FlowMsg) -> None
```

## 参数说明

flowfunc\_wrapper模块中定义的FlowMsg。实际执行时由C++代码传入，通过pybind11的绑定关系映射成flowfunc\_wrapper的FlowMsg对象。

## 返回值

返回FlowMsg类型的对象。

## 异常处理

无

## 约束说明

无
