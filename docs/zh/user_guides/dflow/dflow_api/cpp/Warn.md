# Warn

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

记录Warn级别日志。

## 函数原型

```cpp
virtual void Warn(const char *fmt, ...) __attribute__((format(printf, 2, 3))) = 0
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| fmt | 输入 | 日志格式化控制字符串。<br>取值范围：非空。 |
| ... | 输入 | 可选参数<br>与fmt匹配的参数。 |

## 返回值

无。

## 异常处理

日志流控后日志不会记录。

## 约束说明

无。
