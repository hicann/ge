# E10030 Invalid\_Argument

## 错误信息

报错格式如下，占位符%s的含义依次为属性名称、算子名称：

```text
There is an invalid value for attribute name %s of Op %s in the file specified by --singleop.
```

报错示例如下：

```text
There is an invalid value for attribute name datatype of Op Add in the file specified by --singleop.
```

## 解决方法

检查文件中算子属性值是否有效。
