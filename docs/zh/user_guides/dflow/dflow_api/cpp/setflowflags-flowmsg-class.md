# SetFlowFlags（FlowMsg类）

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

```cpp
void SetFlowFlags(uint32_t flags)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| flags | 输入 | 消息头中的flags标志。<br>flags可以取如下枚举值：<br>enum class FlowFlag : uint32_t {<br>   FLOW_FLAG_EOS = (1U << 0U),  // 数据流结束标志<br>   FLOW_FLAG_SEG = (1U << 1U)  // 非连续数据的分段标志<br>}; |

## 返回值

无。

## 异常处理

无。

## 约束说明

无。
