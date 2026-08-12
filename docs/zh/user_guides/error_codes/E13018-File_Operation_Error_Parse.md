# E13018 File\_Operation\_Error\_Parse

## 错误信息

报错格式如下，占位符%s表示文件名称：

```text
Failed to parse file %s through google::protobuf::TextFormat::Parse.
```

报错示例如下：

```text
Failed to parse file /home/file.prototxt through google::protobuf::TextFormat::Parse.
```

## 可能原因

该文件可能不是有效的Protobuf格式。

## 解决方法

请检查Protobuf文件格式。
