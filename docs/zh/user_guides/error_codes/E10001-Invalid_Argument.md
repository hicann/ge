# E10001 Invalid\_Argument

## 错误信息

报错格式如下，占位符%s的含义依次为参数值、参数名、报错原因：

```text
Value %s for parameter %s is invalid. Reason: %s
```

报错示例1如下：

```text
Value 2 for parameter ge.exec.enableDump is invalid. Reason: The value must be 1 or 0.
```

报错示例2如下：

```text
Value -1 for parameter ge.exec.hostSchedulingMaxThreshold is invalid. Reason: The current value is not within the valid range. The valid range is [0, INT64_MAX].
```

报错示例3如下：

```text
Value FORMAT_ALL for parameter --input_format is invalid. Reason: The current value is not within the valid range.
```

## 解决方法

需按照Reason中的提示输入正确的参数值，或参考官方文档查看相关参数的使用说明。
