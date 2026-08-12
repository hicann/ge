# E10046 Invalid\_Argument\_Tensor\_Dynamic\_Shape

## 错误信息

```text
The total number of -1 axes in the --input_shape argument is greater than the dimension count per profile in --dynamic_dims.
```

## 解决方法

确保--input\_shape参数中-1轴的总数小于--dynamic\_dims中每个档位的维度数量。
