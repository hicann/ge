# E11018 Invalid\_Argument\_Caffe\_Model\_Data

## 错误信息

报错格式如下，占位符%s表示算子名称：

```text
Op name %s contains invalid characters.
```

报错示例如下：

```text
Op name add_&* contains invalid characters.
```

## 解决方法

允许的字符包括：字母、数字、连字符（-）、句号（。）、下划线（\_）和斜杠（/），请修改算子名称后重试。
