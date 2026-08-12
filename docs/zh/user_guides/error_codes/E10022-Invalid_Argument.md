# E10022 Invalid\_Argument

## 错误信息

报错格式如下，占位符%s的含义依次为路径、参数名：

```text
Path %s for parameter --%s does not include the file name.
```

报错示例如下：

```text
Path / for parameter --output does not include the file name.
```

## 解决方法

将文件名添加到路径中。
