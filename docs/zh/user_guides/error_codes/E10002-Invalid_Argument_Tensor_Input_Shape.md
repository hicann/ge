# E10002 Invalid\_Argument\_Tensor\_Input\_Shape

## 错误信息

报错格式如下，占位符%s的含义依次为参数值、报错原因、配置示例：

```text
Value %s for parameter --input_shape is invalid. Reason: %s. The value must be formatted as %s.
```

报错示例1如下：

```text
Value n1~n2,c1,h1,w1 for parameter --input_shape is invalid. Reason: The shape must contain two parts: name and value. The value must be formatted as "input_name1:n1~n2,c1,h1,w1".
```

报错示例2如下：

```text
Value input_name1:1.1,3,224,224 for parameter --input_shape is invalid. Reason: The float number is unsupported. The value must be formatted as "input_name1:1,3,224,224".
```

## 解决方法

--input\_shape参数值的有效格式为：input\_name1:n1,c1,h1,w1;input\_name2:n2,c2,h2,w2。其中，input\_name替换为节点名称。请确保Shape值均为整数。
