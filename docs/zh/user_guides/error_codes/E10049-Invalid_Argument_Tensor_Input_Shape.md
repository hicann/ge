# E10049 Invalid\_Argument\_Tensor\_Input\_Shape

## 错误信息

报错格式如下，占位符%s都表示维度数：

```text
Dimension count %s configured in --input_shape does not match dimension count %s of the node.
```

报错示例如下：

```text
Dimension count 3 configured in --input_shape does not match dimension count 4 of the node.
```

## 解决方法

根据节点的维度数量，在--input\_shape中设置相应的维度数量。
