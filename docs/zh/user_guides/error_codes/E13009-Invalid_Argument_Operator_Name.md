# E13009 Invalid\_Argument\_Operator\_Name

## 错误信息

报错格式如下，占位符%s表示算子名称：

```text
Operator %s already exists in the graph. Ensure that operator names are unique.
```

报错示例如下：

```text
Operator add already exists in the graph. Ensure that operator names are unique.
```

## 解决方法

确保graph中的算子名称唯一。
