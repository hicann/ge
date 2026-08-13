# E13023 Invalid\_Argument\_OM\_Model\_Size

## 错误信息

报错格式如下，占位符%s的含义依次为模型属性名、内存大小、最大值：

```text
Model %s has a size of %s bytes, which exceeds system limit of %s bytes.
```

报错示例如下：

```text
Model tiling data has a size of 4294967298 bytes, which exceeds system limit of 4294967295 bytes.
```

## 可能原因

生成的OM模型过大，无法转储到磁盘。

## 解决方法

请减小模型大小。
