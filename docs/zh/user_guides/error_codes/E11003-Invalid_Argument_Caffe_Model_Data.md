# E11003 Invalid\_Argument\_Caffe\_Model\_Data

## 错误信息

报错格式如下，占位符%s的含义依次为输入维度大小、输入数量：

```text
The number of input_dim fields in the model is %s, which is not 4x the input count %s.
```

报错示例如下：

```text
The number of input_dim fields in the model is 4, which is not 4x the input count 8.
```

## 解决方法

修改Caffe模型并重试。
