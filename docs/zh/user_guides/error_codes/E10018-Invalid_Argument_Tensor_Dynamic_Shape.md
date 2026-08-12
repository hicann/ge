# E10018 Invalid\_Argument\_Tensor\_Dynamic\_Shape

## 错误信息

报错格式如下，占位符%s的含义依次为Shape、index：

```text
Value %s for shape %s is invalid. When --dynamic_batch_size is included, only batch size N can be -1 in --input_shape.
```

报错示例如下：

```text
Value -1 for shape 1 is invalid. When --dynamic_batch_size is included, only batch size N can be -1 in --input_shape.
```

## 解决方法

请使用有效的--input\_shape参数值重试，确保除batch size之外，其他维度的值不为-1。
