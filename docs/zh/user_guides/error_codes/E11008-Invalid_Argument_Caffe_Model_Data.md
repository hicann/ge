# E11008 Invalid\_Argument\_Caffe\_Model\_Data

## 错误信息

```text
Op type DetectionOutput is unsupported.
```

## 解决方法

修改Caffe模型，将DetectionOutput算子替换为FSRDetectionOutput或SSDDetectionOutput。
