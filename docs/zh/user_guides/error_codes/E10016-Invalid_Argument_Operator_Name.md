# E10016 Invalid\_Argument\_Operator\_Name

## 错误信息

报错格式如下，占位符%s的含义依次为算子名称、参数名：

```text
Op name %s specified in --%s is not found in the model. Confirm whether this node name exists, or whether the node is not split with the specified delimiter ';'.
```

报错示例如下：

```text
Op name invalid_op specified in --input_shape is not found in the model. Confirm whether this node name exists, or whether the node is not split with the specified delimiter ';'.
```

## 解决方法

指定Graph中已有节点的名称。
