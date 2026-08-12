# E10034 Invalid\_Argument

## 错误信息

报错格式如下，占位符%s表示算子名称：

```text
Nodes (for example, %s) connected to AIPP must not be of type FP16.
```

报错示例如下：

```text
Nodes (for example, Add) connected to AIPP must not be of type FP16.
```

## 解决方法

- 若要启用AIPP（Artificial Intelligence Pre-Processing，人工智能预处理）功能，需从--input\_fp16\_nodes参数中删除与AIPP连接的节点。
- 若无需启用AIPP功能 ，可atc命令中移除--insert\_op\_conf参数。
