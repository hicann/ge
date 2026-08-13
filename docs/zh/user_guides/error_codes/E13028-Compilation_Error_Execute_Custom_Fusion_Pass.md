# E13028 Compilation\_Error\_Execute\_Custom\_Fusion\_Pass

## 错误信息

报错格式如下，占位符%s的含义依次为融合规则名称、返回码、报错原因：

```text
Failed to run custom fusion pass %s. Return code: %s. Reason: %s.
```

用户自定义的融合规则，规则名称、返回码、报错原因也是自定义的，因此报错示例需以用户自定义的场景为准。

## 解决方法

检查错误日志以获取详细信息，并验证融合逻辑是否正确。
