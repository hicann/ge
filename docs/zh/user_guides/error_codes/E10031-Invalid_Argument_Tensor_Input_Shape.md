# E10031 Invalid\_Argument\_Tensor\_Input\_Shape

## 错误信息

```text
--dynamic_batch_size is included, but none of the nodes specified in --input_shape has a batch size equaling -1.
```

## 可能原因

由于命令行中包含了--dynamic\_batch\_size参数，请确保在--input\_shape参数中指定的节点中至少有一个节点的batch size等于-1。

## 解决方法

1. 在静态Shape场景下，从命令行中移除--dynamic\_batch\_size参数。
2. 在动态shape场景下，需要将--input\_shape中动态shape输入的对应轴设置为-1。
