# ToFlowMsg（tensor）

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

根据输入的Tensor转换成用于承载Tensor的FlowMsg。

## 函数原型

```cpp
FlowMsgPtr ToFlowMsg(const Tensor &tensor)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| tensor | 输入 | Tensor对象。 |

## 返回值

转换的FlowMsg指针。

## 异常处理

转换失败则返回NULL。

## 约束说明

无。
