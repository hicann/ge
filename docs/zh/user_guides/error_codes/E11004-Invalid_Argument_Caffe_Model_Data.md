# E11004 Invalid\_Argument\_Caffe\_Model\_Data

## 错误信息

报错格式如下，占位符%s的含义依次为输入Shape大小、输入数量：

```text
The number of input shapes is %s, which does not match the number of inputs %s.
```

报错示例如下：

```text
The number of input shapes is 3, which does not match the number of inputs 4.
```

## 解决方法

修改Caffe模型并重试。
