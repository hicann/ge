# AllocRawDataMsg（FlowBufferFactory数据类型）

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

根据输入的size申请一块连续内存，用于承载raw data类型的数据。

## 函数原型

```cpp
FlowMsgPtr AllocRawDataMsg(size_t size, uint32_t align = 512U)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| size | 输入 | 申请内存大小。 |
| align | 输入 | 申请内存地址对齐大小，取值范围【32、64、128、256、512、1024】。<br>当前为预留参数，不进行参数值校验。 |

## 返回值

申请的FlowMsg指针。

## 异常处理

申请不到FlowMsg指针则返回NULL。

## 约束说明

无。
