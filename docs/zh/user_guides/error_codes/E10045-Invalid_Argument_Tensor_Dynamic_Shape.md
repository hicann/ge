# E10045 Invalid\_Argument\_Tensor\_Dynamic\_Shape

## 错误信息

```text
The number of -1 axes in the --input_shape argument exceeds the dimension count per profile in --dynamic_dims.
```

## 解决方法

确保--input\_shape参数中-1的数量与--dynamic\_dims中每个档位的维度数量相匹配。
