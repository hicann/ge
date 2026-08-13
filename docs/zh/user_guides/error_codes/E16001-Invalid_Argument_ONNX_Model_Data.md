# E16001 Invalid\_Argument\_ONNX\_Model\_Data

## 错误信息

报错格式如下，占位符%s表示节点名称：

```text
The model has no %s node.
```

报错示例如下：

```text
The model has no input node.
```

## 解决方法

检查ONNX模型是否包含输入节点。
