# E11027 Invalid\_Argument\_Caffe\_Model\_Data

## 错误信息

报错格式如下，占位符%s的含义依次为算子名称、算子类型：

```text
Op %s with optype %s in the Caffe model has an input node with shape size 0.
```

报错示例如下：

```text
Op add with optype Add in the Caffe model has an input node with shape size 0.
```

## 解决方法

无效的Caffe模型，请修改节点的输入Shape。
