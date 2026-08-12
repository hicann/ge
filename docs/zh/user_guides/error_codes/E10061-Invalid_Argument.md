# E10061 Invalid\_Argument

## 错误信息

报错格式如下，占位符%s分别表示参数值、参数名、期望值：

```text
Value %s for parameter %s is invalid. Expected value: %s.
```

报错示例如下：

```text
Value NZ for parameter input_format is invalid. Expected value: ND, NCHW, NHWC, CHWN, NC1HWC0, or NHWC1C0.
```

## 解决方法

需按照报错提示输入正确的参数值。
