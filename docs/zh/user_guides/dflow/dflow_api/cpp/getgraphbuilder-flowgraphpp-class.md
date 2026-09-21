# GetGraphBuilder（FlowGraphPp类）

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

获取FlowGraphPp中Graph的创建函数。

## 函数原型

```cpp
GraphBuilder GetGraphBuilder() const
```

## 参数说明

无

## 返回值

Graph的创建函数，函数原型如下。

```cpp
std::function<ge::Graph()>
```

## 异常处理

无。

## 约束说明

无。
