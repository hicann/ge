# GetProcessPointType

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

获取ProcessPoint的类型。

## 函数原型

```cpp
ProcessPointType GetProcessPointType() const
```

## 参数说明

无

## 返回值

返回一个ProcessPoint的类型。类型取值如下：

```cpp
enum class ProcessPointType {
FUNCTION = 0,
GRAPH = 1,
INNER = 2,
FLOW_GRAPH = 3,
INVALID = 4,
};
```

## 异常处理

无。

## 约束说明

无。
