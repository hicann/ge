# E13010 Not\_Supported\_Operator

## 错误信息

报错格式如下，占位符%s的含义依次为算子名称、算子类型：

```text
No operator plugin is registered for Op: %s, optype: %s.
```

报错示例如下：

```text
No operator plugin is registered for Op: acustom_op, optype: CustomOp.
```

## 解决方法

- 如果算子是自定义算子，则需要注册相关交付件。
- 如果算子是内置算子，则需要安装支持该算子的版本包。
