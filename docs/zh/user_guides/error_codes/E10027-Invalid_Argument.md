# E10027 Invalid\_Argument

## 错误信息

报错格式如下，占位符%s的含义依次为属性名称、输入或输出、tensor index、算子名称：

```text
Attribute %s of %s tensor %s for Op %s is invalid when --singleop is specified.
```

报错示例如下：

```text
Attribute datatype of input tensor 1 for Op Add is invalid when --singleop is specified.
```

## 解决方法

请使用有效的tensor dtype和format。
