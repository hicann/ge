# E12004 Invalid\_Argument\_Operator\_Input\_Index

## 错误信息

报错格式如下，占位符%s的含义依次为算子名称、input index、input数量：

```text
Failed to register the prototype of Op %s. If input index is less than 0, then input index -%s (absolute value) must be less than the input count %s.
```

报错示例如下：

```text
Failed to register the prototype of Op add. If input index is less than 0, then input index -2 (absolute value) must be less than the input count 1.
```

## 解决方法

当Const输入被转换为属性时，检查输入索引是否设置正确。
