# E11009 Not\_Supported\_Operator

## 错误信息

报错格式如下，占位符%s的含义依次为算子名称、算子类型：

```text
No Caffe parser is registered for Op %s with Op type %s.
```

报错示例如下：

```text
No Caffe parser is registered for Op custom_op with Op type CustomOp.
```

## 解决方法

检查算子的Caffe插件是否已注册。
