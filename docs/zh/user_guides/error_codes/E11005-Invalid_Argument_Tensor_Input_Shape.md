# E11005 Invalid\_Argument\_Tensor\_Input\_Shape

## 错误信息

报错格式如下，占位符%s表示输入名称：

```text
The shape is not defined by using --input_shape for input %s.
```

报错示例如下：

```text
The shape is not defined by using --input_shape for input Input1.
```

## 解决方法

修改Caffe模型，或者在atc命令行中将输入Shape添加到--input\_shape参数中。
