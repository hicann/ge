# E10009 Invalid\_Argument\_Tensor\_Dynamic\_Shape

## 错误信息

```text
--dynamic_batch_size, --dynamic_image_size, --input_shape_range, and --dynamic_dims are mutually exclusive.
```

## 解决方法

1. 在动态Shape场景中，请在命令行中仅包含这些参数中的一个。
2. 在静态Shape场景中，请从命令行中移除这些参数。
