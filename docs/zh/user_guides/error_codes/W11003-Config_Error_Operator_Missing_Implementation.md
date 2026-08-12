# W11003 Config\_Error\_Operator\_Missing\_Implementation

## 错误信息

报错格式如下，占位符%s的含义依次为算子名称、交付件名称：

```text
Operator %s lacks required %s implementation.
```

报错示例如下：

```text
Operator CustomAdd lacks required InferShape implementation.
```

## 解决方法

算子实现不完整，确保提供所有算子所需的实现（例如，tiling），详细信息请参见官方的算子开发文档。
