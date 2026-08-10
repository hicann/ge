# GEStreamAllocationSummaryGetAllNodes

## 产品支持情况

请参见[Session接口产品支持情况](../../cpp/ge/Session/overview.md)。

## 头文件/库文件

- 头文件：\#include <ge/ge\_graph\_compile\_summary.h\>
- 库文件：libge\_compiler.so

## 功能说明

获取根图和子图的所有节点。

## 函数原型

```c
ge::Status GEStreamAllocationSummaryGetAllNodes(const ge::CompiledGraphSummary &compiled_graph_summary, std::map<AscendString, std::vector<std::vector<GNode>>> &graph_to_all_nodes);
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| compiled_graph_summary | 输入 | 图编译后的概要信息。 |
| graph_to_all_nodes | 输出 | map格式，key为图名称，value为所有节点的向量，其中索引表示逻辑流ID。 |

## 返回值说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| - | ge::Status | - SUCCESS：接口调用成功。<br>  - FAILED：接口调用失败 |

## 约束说明

无
