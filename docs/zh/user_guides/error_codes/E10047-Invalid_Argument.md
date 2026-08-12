# E10047 Invalid\_Argument

## 错误信息

报错格式如下，占位符%s都表示参数名：

```text
--%s and --%s are mutually exclusive.
```

报错示例如下：

```text
--enable_compress_weight and --compress_weight_conf are mutually exclusive.
```

## 解决方法

删除其中一个参数后再重试。
