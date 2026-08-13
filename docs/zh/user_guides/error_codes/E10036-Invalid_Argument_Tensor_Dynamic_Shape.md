# E10036 Invalid\_Argument\_Tensor\_Dynamic\_Shape

## 错误信息

报错格式如下，占位符%s的含义依次为Shape size、Shape size最大值：

```text
--dynamic_batch_size, --dynamic_image_size, or --dynamic_dims has %s profiles, which is greater than the maximum %s.
```

报错示例如下：

```text
--dynamic_batch_size, --dynamic_image_size, or --dynamic_dims has 1024 profiles, which is greater than the maximum 100.
```

## 解决方法

确保在--dynamic\_batch\_size、--dynamic\_image\_size或--dynamic\_dims中配置的档位数量不超过最大值。
