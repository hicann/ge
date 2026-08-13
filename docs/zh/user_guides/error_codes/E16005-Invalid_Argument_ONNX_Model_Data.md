# E16005 Invalid\_Argument\_ONNX\_Model\_Data

## 错误信息

报错格式如下，占位符%s表示domain版本数量：

```text
The model has %s --domain_version fields, but only one is allowed.
```

报错示例如下：

```text
The model has 2 --domain_version fields, but only one is allowed.
```

## 解决方法

无效的ONNX模型。请修改ONNX模型，如果算子节点上没有指定domain，则模型上只能指定一个domain。
