# E11033 Invalid\_Argument\_Caffe\_Model\_Weight

## 错误信息

报错格式如下，占位符%s的含义依次为blob名称、blob大小、报错原因：

```text
Failed to convert the weight file. Blob %s of size %s is invalid. Reason: %s.
```

报错示例如下：

```text
Failed to convert the weight file. Blob data of size 100 is invalid. Reason: It does not match shape size 128.
```

## 可能原因

Caffe权重文件中节点的blob大小与根据其Shape计算出的元素数量不匹配。

## 解决方法

请尝试使用有效的Caffe模型或权重文件，确保这两个文件相互匹配。
