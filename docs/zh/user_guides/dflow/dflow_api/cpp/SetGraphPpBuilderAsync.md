# SetGraphPpBuilderAsync

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

设置FlowGraph中的GraphPp的Builder是否异步执行。

## 函数原型

```cpp
void SetGraphPpBuilderAsync(bool graphpp_builder_async)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| graphpp_builder_async | 输入 | FlowGraph中的GraphPp的Builder是否异步执行。取值如下：<br><br>  - true：是<br>  - false：否<br><br>默认值：false |

## 返回值

无

## 异常处理

无。

## 约束说明

无。
