# E13025 Invalid\_Argument

## 错误信息

报错格式如下，占位符%s表示报错原因：

```text
Input tensor is invalid. Reason: %s.
```

报错示例1如下：

```text
Input tensor is invalid. Reason: Data indexes must be consecutive and start from 0 when the data shape range is enabled.
```

报错示例2如下：

```text
Input tensor is invalid. Reason: The number of inputs/outputs provided by the user is inconsistent with that required by the model.
```

## 解决方法

需按照Reason中的提示输入正确的参数值，或参考官方文档查看相关参数的使用说明。
