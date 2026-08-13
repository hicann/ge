# E10003 Invalid\_Argument

## 错误信息

报错格式如下，占位符%s的含义依次为参数值、参数名、报错原因：

```text
Value %s for parameter --%s is invalid. Reason: %s
```

报错示例1如下：

```text
Value 1.1,2,4,8 for parameter --dynamic_batch_size is invalid. Reason: It can only contain digits and ",".
```

## 解决方法

需按照Reason中的提示输入正确的参数值，或参考官方文档查看相关参数的使用说明。
