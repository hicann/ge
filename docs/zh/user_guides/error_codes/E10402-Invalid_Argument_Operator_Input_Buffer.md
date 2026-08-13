# E10402 Invalid\_Argument\_Operator\_Input\_Buffer

## 错误信息

报错格式如下，占位符%s的含义依次为index、期望buffer大小、实际buffer大小：

```text
Input indexed %s requires a %s buffer, but %s aligned buffer is allocated.
```

报错示例如下：

```text
Input indexed 1 requires a 200 buffer, but 100 aligned buffer is allocated.
```

## 解决方法

检查数据类型、维度和Shape是否设置正确，具体请参见官方文档中的aclGetTensorDescSize接口说明。
