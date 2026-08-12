# E10062 Invalid\_Argument\_API\_Call\_Sequence

## 错误信息

报错格式如下，占位符%s分别表示接口名、报错原因：

```text
Failed to %s. Reason: %s.
```

报错示例如下：

```text
Failed to call RunGraphAsync. Reason: Graph <graph_id> has been compiled by calling CompileGraph. RunGraphAsync and CompileGraph are mutually exclusive and cannot be used together.
```

## 解决方法

根据Reason中的提示调整代码。
