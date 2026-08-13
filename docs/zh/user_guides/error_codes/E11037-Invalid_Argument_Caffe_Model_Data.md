# E11037 Invalid\_Argument\_Caffe\_Model\_Data

## 错误信息

报错格式如下，占位符%s表示算子名称：

```text
Op %s has zero outputs.
```

报错示例如下：

```text
Op add has zero outputs.
```

## 解决方法

Caffe模型中的节点必须至少有一个输出。
