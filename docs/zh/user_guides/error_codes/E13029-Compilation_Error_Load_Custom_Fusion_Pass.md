# E13029 Compilation\_Error\_Load\_Custom\_Fusion\_Pass

## 错误信息

报错格式如下，占位符%s的含义依次为融合规则库、报错原因：

```text
Failed to load custom fusion pass lib %s. Reason: %s.
```

报错示例如下：

```text
Failed to load custom fusion pass lib /custom_op.so. Reason: undefined symbol: _ZNK7c10_npu9NPUStream6streamEv.
```

## 解决方法

分析上述提到的失败原因。以下是一些常见的dlopen失败情况的典型解决方案：

1. 确认库路径正确且文件存在。
2. 确保库及其依赖项具有正确的权限。
3. 使用'ldd'命令检查所有依赖项是否可用。
