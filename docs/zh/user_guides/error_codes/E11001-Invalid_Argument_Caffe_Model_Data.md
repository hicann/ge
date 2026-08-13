# E11001 Invalid\_Argument\_Caffe\_Model\_Data

## 错误信息

```text
input_dim and input_shape are mutually exclusive in NetParameter for Caffe model conversion.
```

## 解决方法

从atc命令行中删除--input\_dim或--input\_shape参数。
