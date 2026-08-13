# E12009 Invalid\_Argument\_TensorFlow\_Model\_Data

## 错误信息

报错格式如下，占位符%s的含义依次为输入名称、算子名称：

```text
Input %s for Op %s is not found in graph_def.
```

报错示例如下：

```text
Input data for Op input is not found in graph_def.
```

## 可能原因

graph中未找到节点的输入名称。

## 解决方法

请使用有效的TensorFlow模型重试。
