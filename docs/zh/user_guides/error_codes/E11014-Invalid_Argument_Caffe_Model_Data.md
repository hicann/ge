# E11014 Invalid\_Argument\_Caffe\_Model\_Data

## 错误信息

报错格式如下，占位符%s表示layer：

```text
Failed to find the top blob for layer %s.
```

报错示例如下：

```text
Failed to find the top blob for layer conv1.
```

## 可能原因

top blob在源Caffe模型中没有对应的节点。

## 解决方法

修改Caffe模型并重试。
