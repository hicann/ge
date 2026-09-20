# registered

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

判断消息类型ID是否被注册过

## 函数原型

```python
registered(msg_type)
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| msg_type | int | 注册的类型ID。 |

## 返回值

bool，True表示注册过，False表示没有注册

## 调用示例

```python
import dataflow as df
registered = df.msg_type_register.registered(1026)
```

## 约束说明

无
