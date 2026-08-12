# E10025 Invalid\_Argument

## 错误信息

报错格式如下，占位符%s的含义依次为文件名、报错原因：

```text
File %s specified by --singleop is not a valid JSON file. Reason: %s.
```

报错示例如下：

```text
File /home/singleop.json specified by --singleop is not a valid JSON file. Reason: ios_base::clear: unspecified iostream_category error.
```

## 解决方法

检查文件是否为有效的JSON格式。
