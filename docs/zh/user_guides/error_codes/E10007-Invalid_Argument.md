# E10007 Invalid\_Argument

## 错误信息

报错格式如下，占位符%s的含义依次为参数名、期望值：

```text
--%s is required. The value must be %s.
```

报错示例如下：

```text
--framework is required. The value must be 0(Caffe) or 1(MindSpore) or 3(TensorFlow) or 5(Onnx).
```

## 解决方法

请设置有效的参数值。
