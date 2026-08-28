# ReportMatch

## 产品支持情况

全量芯片支持。

## 头文件/库文件

- 头文件：\#include <ge/fusion/graph\_fuse\_inspector\_utils.h\>
- 库文件：libgraph\_base.so

## 功能说明

上报一次结构匹配，在图遍历中发现目标结构后调用，无论融合条件是否通过均计入。内部自动累加 match\_time，不改变 effect\_time，对应信息落盘至fusion\_result.json。

与 [ReportFuse](ReportFuse.md) 配合使用可统计结构匹配的命中率：match\_time 为结构匹配总次数（全集），effect\_time 为融合实际生效次数（子集），match\_time - effect\_time 反映因条件过滤而放弃融合的数量。

## 函数原型

```c++
static Status ReportMatch(const std::vector<GNode> &matched_nodes, CustomPassContext &ctx)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| matched_nodes | 输入 | 结构匹配命中的节点列表（列表内所有节点需连通）。 |
| ctx | 输入 | Pass上下文，使用ctx.GetPassName()记录pass name。 |

## 返回值说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| - | Status | SUCCESS：上报成功<br>FAILED：上报失败 |

## 约束说明

该接口应在发现目标结构后、[CanFuse](CanFuse.md) 之前调用。
