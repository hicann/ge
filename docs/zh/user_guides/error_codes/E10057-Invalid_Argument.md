# E10057 Invalid\_Argument

## 错误信息

报错格式如下，占位符%s都表示参数名：

```text
--%s and --%s can only be used together.
```

报错示例如下：

```text
--om and --model can only be used together.
```

## 解决方法

如果--mode的值为6，则只能与--om一起使用，请检查并重试。
