# E10038 Invalid\_Argument\_Tensor\_Dynamic\_Shape

## 错误信息

报错格式如下，占位符%s表示维度值：

```text
Dimension size %s is invalid. The value must be greater than 0.
```

报错示例如下：

```text
Dimension size -1 is invalid. The value must be greater than 0.
```

## 解决方法

在--dynamic\_batch\_size、--dynamic\_image\_size或--dynamic\_dims参数值中，将每个档位的值设置为正值。
