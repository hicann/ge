# E10029 Invalid\_Argument

## 错误信息

报错格式如下，占位符%s表示算子名称：

```text
Attribute name of Op %s is empty in the file specified by --singleop.
```

报错示例如下：

```text
Attribute name of Op Add is empty in the file specified by --singleop.
```

## 解决方法

检查文件中算子属性名称是否为空。
