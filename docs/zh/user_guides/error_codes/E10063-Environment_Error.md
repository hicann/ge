# E10063 Environment\_Error

## 错误信息

报错格式如下，占位符%s分别表示接口名、报错原因：

```text
Failed to call the %s API of the system or third-party software. Reason: %s.
```

报错示例如下：

```text
Failed to call the localtime API of the system or third-party software. Reason: [Errno 75] Value too large for defined data type.
```

## 解决方法

根据Reason中的提示调整代码。
