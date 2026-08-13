# E10020 Invalid\_Argument

## 错误信息

报错格式如下，占位符%s表示参数值：

```text
Value %s for parameter --dynamic_image_size is invalid.
```

报错示例如下：

```text
Value 1,2,3;4,5,6 for parameter --dynamic_image_size is invalid.
```

## 解决方法

该值必须格式化为"imagesize1\_height,imagesize1\_width;imagesize2\_height,imagesize2\_width"，请确保每个配置项都有两个维度。
