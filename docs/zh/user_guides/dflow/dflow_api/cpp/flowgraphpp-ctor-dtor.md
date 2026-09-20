# FlowGraphPp构造函数和析构函数

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

FlowGraphPp构造函数和析构函数，构造函数会返回一个FlowGraphPp对象。

## 函数原型

```cpp
FlowGraphPp(const char_t *pp_name, const FlowGraphBuilder &builder)
~FlowGraphPp() override
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| pp_name | 输入 | FlowGraphPp的名称，需要全图唯一。 |
| builder | 输入 | FlowGraph的构建方法：std::function<FlowGraph()><br>FlowGraph的构建具体请参考[FlowGraph类](flowgraph-class.md)。 |

## 返回值

返回一个FlowGraphPp对象。

## 异常处理

无。

## 约束说明

无。
