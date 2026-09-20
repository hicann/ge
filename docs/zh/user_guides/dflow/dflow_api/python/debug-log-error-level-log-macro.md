# 调试日志Error级别日志宏

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

调试日志Error级别日志宏。

## 函数原型

```python
error(self, message: str, *args: tuple) -> None
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| message | 输入 | 日志格式化控制字符串。<br>取值范围：非空。 |
| args | 输入 | 可选参数<br>与message匹配的参数。 |

## 返回值

无

## 异常处理

无

## 约束说明

无
