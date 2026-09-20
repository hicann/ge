# FlowNode构造函数和析构函数

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

FlowNode构造函数和析构函数，构造函数返回一个FlowNode节点。

## 函数原型

```cpp
FlowNode(const char *name, uint32_t input_num, uint32_t output_num)
~FlowNode() override
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| name | 输入 | 计算节点名称，需要全图唯一。 |
| input_num | 输入 | 节点的输入个数。 |
| output_num | 输入 | 节点的输出个数。 |

## 返回值

返回一个FlowNode节点。

## 异常处理

无。

## 约束说明

无。
