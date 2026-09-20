# SetContainsNMappingNode

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

设置FlowGraph是否包含n\_mapping节点。n\_mapping节点指存在以下任意一种情况。

- 一次输入生成非一次（多个，0个，或不定）输出。
- 多次输入生成一次输出。
- 多次输入生成非一次（多个，0个，或不定）输出。

## 函数原型

```cpp
FlowGraph &SetContainsNMappingNode(bool contains_n_mapping_node)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| contains_n_mapping_node | 输入 | 是否包含n_mapping节点。<br>图中只要有一个n_mapping处理节点，则设置为true,默认为false。 |

## 返回值

返回设置了是否包含n\_mapping节点属性的FlowGraph图。

## 异常处理

无。

## 约束说明

无。
