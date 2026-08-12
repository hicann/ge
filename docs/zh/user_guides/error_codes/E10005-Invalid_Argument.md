# E10005 Invalid\_Argument

## 错误信息

报错格式如下，占位符%s的含义依次为参数值、参数名：

```text
Value %s for parameter --%s is invalid. The value must be either true or false.
```

报错示例如下：

```text
Value enable for parameter --is_input_adjust_hw_layout is invalid. The value must be either true or false.
```

## 解决方法

请设置有效的参数值，参数值只能是true或false。
