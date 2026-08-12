# E13024 Config\_Error\_Invalid\_Environment\_Variable

## 错误信息

报错格式如下，占位符%s的含义依次为环境变量名称、环境变量取值、报错原因：

```text
Value %s for environment variable %s is invalid. Reason: %s.
```

报错示例如下：

```text
Value 1 for environment variable VIRTUAL_TYPE is invalid. Reason: L1_fusion is not supported in the Ascend virtual instance scenario.
```

## 解决方法

请参见环境变量参考文档重新设置环境变量。
