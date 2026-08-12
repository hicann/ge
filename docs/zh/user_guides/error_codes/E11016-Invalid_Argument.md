# E11016 Invalid\_Argument

## 错误信息

报错格式如下，占位符%s的含义依次为算子名称、output index、output index最大值、input index、input index最大值：

```text
Failed to add Op %s to NetOutput. Op output index %s is not less than %s. NetOutput input_index %s is not less than %s.
```

报错示例如下：

```text
Failed to add Op add to NetOutput. Op output index 3 is not less than 2. NetOutput input_index 3 is not less than 2.
```

## 解决方法

请使用有效的--out\_nodes参数重试。
