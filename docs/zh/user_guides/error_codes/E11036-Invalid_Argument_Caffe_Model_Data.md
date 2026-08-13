# E11036 Invalid\_Argument\_Caffe\_Model\_Data

## 错误信息

报错格式如下，占位符%s表示top blobs：

```text
Data nodes have duplicate top blobs %s.
```

报错示例如下：

```text
Data nodes have duplicate top blobs data1.
```

## 解决方法

无效的Caffe模型。请确保data节点名称唯一。
