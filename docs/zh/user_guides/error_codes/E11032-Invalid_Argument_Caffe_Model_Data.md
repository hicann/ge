# E11032 Invalid\_Argument\_Caffe\_Model\_Data

## 错误信息

报错格式如下，占位符%s的含义依次为消息类型、错误字段、报错原因：

```text
Failed to parse message %s. The error field is %s. Reason: %s.
```

报错示例如下：

```text
Failed to parse message model. The error field is LayerParameter. Reason: Cannot find domi.caffe.LayerParameter in google::protobuf::Descriptor.
```

## 解决方法

检查Caffe模型是否支持该字段。
