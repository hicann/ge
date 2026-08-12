# E11029 Invalid\_Argument\_Caffe\_Model\_Data

## 错误信息

报错格式如下，占位符%s表示算子名称：

```text
Op %s exists in the model file but is not found in weight file.
```

报错示例如下：

```text
Op add exists in the model file but is not found in weight file.
```

## 解决方法

请尝试使用有效的Caffe模型或权重文件，确保这两个文件相互匹配。
