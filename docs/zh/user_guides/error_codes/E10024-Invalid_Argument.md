# E10024 Invalid\_Argument

## 错误信息

报错格式如下，占位符%s表示文件名：

```text
Failed to open file %s specified by --singleop.
```

报错示例如下：

```text
Failed to open file /home/singleop.json specified by --singleop.
```

## 解决方法

检查文件的用户属组和权限设置，确保运行atc命令的用户具有足够的权限来打开该文件。
