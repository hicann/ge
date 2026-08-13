# E11012 Invalid\_Argument\_Caffe\_Model\_Data

## 错误信息

报错格式如下，占位符%s的含义依次为bottom blob、layer、index：

```text
Unknown bottom blob %s at layer %s. The bottom blob is indexed %s.
```

报错示例如下：

```text
Unknown bottom blob data at layer conv1. The bottom blob is indexed 1.
```

## 解决方法

修改Caffe模型并重试。
