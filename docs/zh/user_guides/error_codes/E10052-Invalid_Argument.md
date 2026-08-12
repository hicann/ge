# E10052 Invalid\_Argument

## 错误信息

报错格式如下，占位符%s表示报错原因：

```text
AIPP configuration is invalid. Reason: %s.
```

报错示例如下：

```text
AIPP configuration is invalid. Reason: When --dynamic_image_size is set, crop and padding cannot be set to 'true'.
```

## 解决方法

需按照Reason中的提示输入正确的参数值，或参考官方文档查看相关参数的使用说明。
