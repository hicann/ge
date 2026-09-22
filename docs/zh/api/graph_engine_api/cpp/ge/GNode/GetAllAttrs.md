# GetAllAttrs

## 产品支持情况

全量芯片支持。

## 头文件/库文件

- 头文件：\#include <graph/gnode.h\>
- 库文件：libgraph.so

## 功能说明

获取Node的全部属性及属性值，包括预定义IR属性和通用属性，不包括输入输出端口属性以及以下划线（`_`）开头的内部属性。
未设置的属性不会写入输出映射。

## 函数原型

```c++
graphStatus GetAllAttrs(std::map<AscendString, AttrValue> &attr_values) const
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| attr_values | 输出 | 返回Node的全部属性及属性值。调用时原有内容会被清空。 |

## 返回值说明

| 类型 | 说明 |
| --- | --- |
| graphStatus | GRAPH_SUCCESS(0)：成功。<br>其他值：失败。 |

## 约束说明

无。

## 调用示例

```c++
std::map<AscendString, AttrValue> attr_values;
node.GetAllAttrs(attr_values);
```
