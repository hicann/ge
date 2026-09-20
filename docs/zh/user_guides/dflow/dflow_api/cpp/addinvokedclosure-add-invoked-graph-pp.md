# AddInvokedClosure \(添加调用的GraphPp\)

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

添加FunctionPp调用的GraphPp，返回添加好的FunctionPp。

## 函数原型

```cpp
FunctionPp &AddInvokedClosure(const char_t *name, const GraphPp &graph_pp)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| name | 输入 | 调用的GraphPp的唯一标识，需要全图唯一。 |
| graph_pp | 输入 | 调用的GraphPp。 |

## 返回值

返回设置好的FunctionPp。

## 异常处理

无。

## 约束说明

无。
