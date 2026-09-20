# GetInvokedClosures

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

获取FunctionPp调用的GraphPp。

## 函数原型

```cpp
const std::map<const std::string, const GraphPp> &GetInvokedClosures() const
```

## 参数说明

无

## 返回值

返回[AddInvokedClosure](addinvokedclosure-add-invoked-graph-pp.md)设置的**<name, graph\_pp\>**的map。

## 异常处理

无。

## 约束说明

无。
