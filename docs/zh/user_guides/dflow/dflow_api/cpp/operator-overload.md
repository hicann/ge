# 关系符重载

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

对于AscendString对象大小比较的使用场景（例如map数据结构的key进行排序），通过重载以下关系符实现。

## 函数原型

```cpp
bool operator<(const AscendString &d) const
bool operator>(const AscendString &d) const
bool operator<=(const AscendString &d) const
bool operator>=(const AscendString &d) const
bool operator==(const AscendString &d) const
bool operator!=(const AscendString &d) const
```

## 参数说明

无。

## 返回值

无。

## 异常处理

无。

## 约束说明

无。
