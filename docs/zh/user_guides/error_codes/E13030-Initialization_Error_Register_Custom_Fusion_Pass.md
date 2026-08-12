# E13030 Initialization\_Error\_Register\_Custom\_Fusion\_Pass

## 错误信息

报错格式如下，占位符%s的含义依次为融合规则名称、报错原因：

```text
Failed to get custom fusion pass func %s. Reason: %s.
```

报错示例如下：

```text
Failed to get custom fusion pass func CustomOpPass. Reason: Custom stream allocation pass function is required in stage AfterBuiltinFusionPass, but got nullptr.
```

## 解决方法

检查自定义融合规则注册是否有效。
