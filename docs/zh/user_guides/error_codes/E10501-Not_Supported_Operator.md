# E10501 Not\_Supported\_Operator

## 错误信息

报错格式如下，占位符%s的含义依次为算子名称、算子类型：

```text
IR for Op %s with optype %s is not registered.
```

报错示例如下：

```text
IR for Op custom_op with optype CustomOp is not registered.
```

## 可能原因

1. 未配置环境变量ASCEND\_OPP\_PATH。
2. 算子IR（Intermediate Representation）未注册。

## 解决方法

1. 请配置该环境变量。
2. 请参考算子开发文档检查算子原型是否已注册。
