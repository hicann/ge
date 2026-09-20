# Serialize（ProcessPoint类）

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

ProcessPoint的序列化方法。由ProcessPoint的子类去实现该方法的功能。

## 函数原型

```cpp
virtual void Serialize(ge::AscendString &str) const = 0
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| str | 输出 | ProcessPoint序列化的字符串。 |

## 返回值

无。

## 异常处理

无。

## 约束说明

无。
