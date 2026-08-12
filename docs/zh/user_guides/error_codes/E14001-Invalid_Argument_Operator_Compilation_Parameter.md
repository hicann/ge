# E14001 Invalid\_Argument\_Operator\_Compilation\_Parameter

## 错误信息

报错格式如下，占位符%s的含义依次为参数值、算子名称、算子类型、报错原因：

```text
Argument %s for Op %s with optype %s is invalid. Reason: %s.
```

报错示例如下：

```text
Argument inputs size 2 for Op add with optype Add is invalid. Reason: Input size is not equal to tensor size.
```

## 解决方法

检查算子的类型、输入和输出是否与配置的参数匹配。
