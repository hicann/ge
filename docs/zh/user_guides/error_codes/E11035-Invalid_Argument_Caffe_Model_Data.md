# E11035 Invalid\_Argument\_Caffe\_Model\_Data

## 错误信息

报错格式如下，占位符%s的含义依次为算子名称、size：

```text
The top size of data node %s is not 1 but %s.
```

报错示例如下：

```text
The top size of data node data1 is not 1 but 2.
```

## 解决方法

无效的Caffe模型，请将data节点的数量更改为1。
