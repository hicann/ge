# AllocEmptyDataMsg（FlowBufferFactory数据类型）

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

申请空数据的MsgType类型的message。

## 函数原型

```cpp
FlowMsgPtr AllocEmptyDataMsg(MsgType type)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| type | 输入 | 要申请空数据的消息类型。 |

## 返回值

申请的FlowMsg指针。

## 异常处理

申请不到FlowMsg指针则返回NULL。

## 约束说明

无。
