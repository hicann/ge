# E11021 Invalid\_Argument\_Caffe\_Model\_Data

## 错误信息

报错格式如下，占位符%s表示模型文件名称：

```text
Model file %s contains "layers" structures, which have been deprecated in Caffe and unsupported by ATC.
```

报错示例如下：

```text
Model file /home/caffe.prototxt contains "layers" structures, which have been deprecated in Caffe and unsupported by ATC.
```

## 解决方法

请用layer替换layers。
