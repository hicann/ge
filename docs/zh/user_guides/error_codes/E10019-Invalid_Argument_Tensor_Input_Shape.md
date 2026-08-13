# E10019 Invalid\_Argument\_Tensor\_Input\_Shape

## 错误信息

```text
When --dynamic_image_size is included, only the height and width axes can be -1 in --input_shape.
```

## 解决方法

请使用有效的--input\_shape参数值重试，确保除高度和宽度之外，其他维度值不为-1。
