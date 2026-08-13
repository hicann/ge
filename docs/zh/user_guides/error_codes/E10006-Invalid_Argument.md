# E10006 Invalid\_Argument

## 错误信息

报错格式如下，占位符%s的含义依次为参数值、参数名：

```text
Value %s for parameter --%s is invalid. The value must be either 1 or 0.
```

报错示例如下：

```text
Value 2 for parameter --sparsity is invalid. The value must be either 1 or 0.
```

## 解决方法

请设置有效的参数值，参数值只能是1或0。
