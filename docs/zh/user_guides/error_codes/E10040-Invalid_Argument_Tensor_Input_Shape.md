# E10040 Invalid\_Argument\_Tensor\_Input\_Shape

## 错误信息

```text
As the --dynamic_batch_size, --dynamic_image_size, or --dynamic_dims argument is included, the corresponding nodes specified in --input_shape must have -1 axes and cannot have '~'.
```

## 解决方法

- 静态shape场景下，从命令行中移除--dynamic\_batch\_size、--dynamic\_image\_size或--dynamic\_dims参数。
- 动态多batch场景下，在--input\_shape参数中将动态shape输入的batch size设置为-1。
- 动态Shape场景下，从命令行中删除--dynamic\_batch\_size、--dynamic\_image\_size或--dynamic\_dims参数，并将--input\_shape设置为-1或设置range值范围。
