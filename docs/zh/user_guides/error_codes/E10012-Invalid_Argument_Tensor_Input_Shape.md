# E10012 Invalid\_Argument\_Tensor\_Input\_Shape

## 错误信息

```text
--dynamic_batch_size is included, but the dimension count of the dynamic-shape input configured in --input_shape is less than 1.
```

## 解决方法

- 在静态Shape场景下，从命令行中移除--dynamic\_batch\_size选项。
- 在动态Shape场景下，将--input\_shape中动态Shape输入的对应轴设置为-1。
