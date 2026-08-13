# E10021 Invalid\_Argument

## 错误信息

报错格式如下，占位符%s的含义依次为参数名、期望值：

```text
Path for parameter --%s is too long. Keep the length within %s.
```

报错示例如下：

```text
Path for parameter --output is too long. Keep the length within 4096.
```

## 解决方法

路径名称超出最大长度，请指定一个有效的路径名称。
