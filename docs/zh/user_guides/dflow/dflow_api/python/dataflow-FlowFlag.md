# dataflow.FlowFlag

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

设置FlowMsg消息头中的flags。

## 函数原型

不涉及

## 参数说明

枚举值如下：

- FlowFlag.DATA\_FLOW\_FLAG\_EOS
- FlowFlag.DATA\_FLOW\_FLAG\_SEG

## 返回值

无

## 调用示例

```python
import dataflow as df
flow_info = df.FlowInfo()
flow_info.flow_flags = df.FlowFlag.DATA_FLOW_FLAG_EOS
```

## 约束说明

无
