# E10048 Invalid\_Argument

## 错误信息

报错格式如下，占位符%s的含义依次为参数值、报错原因、配置示例：

```text
Value %s for parameter --input_shape_range or dynamic_inputs_shape_range is invalid. Reason: %s. The value must be formatted as %s.
```

报错示例如下：

```text
Value abc for parameter --input_shape_range or dynamic_inputs_shape_range is invalid. Reason: The current string cannot be converted to a number. The value must be formatted as 16.
```

## 解决方法

请使用有效的参数值重试。
