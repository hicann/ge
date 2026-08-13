# E10050 Invalid\_Argument\_Tensor\_Input\_Shape\_Range

## 错误信息

报错格式如下，占位符%s的含义依次为当前维度大小、最小值、最大值：

```text
Current dimension size %s is not in the range of %s-%s specified by --input_shape.
```

报错示例如下：

```text
Current dimension size 2 is not in the range of 4-8 specified by --input_shape.
```

## 解决方法

根据--input\_shape参数值设置维度大小。
