# GetDataPos

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

获取输出数据对应权重矩阵中的位置。

## 函数原型

```cpp
const std::vector<std::pair<int32_t, int32_t>> &GetDataPos() const = 0
```

## 参数说明

无。

## 返回值

输出数据对应权重矩阵中的位置， pair中第一个值表示对应的行号，第二个值表示对应的列号。

## 异常处理

无。

## 约束说明

无。
