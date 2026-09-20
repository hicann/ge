# set\_route\_label

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

设置路由的标签。

举例：udf1，udf2节点分别部署在device1和device2，且都有2个输出（out0和out1）。如用户希望每组out0和out1发送到同一个device上，需要给2个输出flowmsg设置相同的route\_label。

## 函数原型

```python
set_route_label(self, route_label) -> None
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| route_label | 输入 | 路由的标签，取值为0表示不使用。 |

## 返回值

无

## 异常处理

无

## 约束说明

无
