# E14002 Invalid\_Argument\_Tensor\_Attribute

## 错误信息

报错格式如下，占位符%s的含义依次为属性名称、报错原因：

```text
In the current process, the attribute of %s must be obtained successfully. Reason: %s.
```

报错示例如下：

```text
In the current process, the attribute of storage_format must be obtained successfully. Reason: Failed to get storage shape from node Failed to get storage shape from node add.
```

## 解决方法

请按照错误提示为算子设置属性信息。
