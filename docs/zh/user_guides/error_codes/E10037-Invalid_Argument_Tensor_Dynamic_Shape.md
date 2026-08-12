# E10037 Invalid\_Argument\_Tensor\_Dynamic\_Shape

## 错误信息

报错格式如下，占位符%s都表示维度数量：

```text
The profiles configured in --dynamic_batch_size, --dynamic_image_size, or --dynamic_dims have inconsistent dimension counts. A profile has %s dimensions while another has %s dimensions.
```

报错示例如下：

```text
The profiles configured in --dynamic_batch_size, --dynamic_image_size, or --dynamic_dims have inconsistent dimension counts. A profile has 4 dimensions while another has 8 dimensions.
```

## 解决方法

确保在--dynamic\_batch\_size、--dynamic\_image\_size或--dynamic\_dims中配置的各档位数据具有相同的维度数量。
