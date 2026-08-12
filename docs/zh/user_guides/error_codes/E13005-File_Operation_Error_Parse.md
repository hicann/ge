# E13005 File\_Operation\_Error\_Parse

## 错误信息

报错格式如下，占位符%s表示文件名称：

```text
Failed to parse file %s.
```

报错示例如下：

```text
Failed to parse file /home/caffe.prototxt.
```

## 解决方法

请检查是否安装了匹配的Protobuf版本，并使用有效文件重试。有关详细信息，请参见官网ATC工具文档中的--framework参数说明。
