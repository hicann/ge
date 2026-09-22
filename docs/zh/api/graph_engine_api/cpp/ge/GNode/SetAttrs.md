# SetAttrs

## 产品支持情况

全量芯片支持。

## 头文件/库文件

- 头文件：\#include <graph/gnode.h\>
- 库文件：libgraph.so

## 功能说明

批量设置Node的属性及属性值。

## 函数原型

```c++
graphStatus SetAttrs(const std::map<AscendString, AttrValue> &attr_values) const
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| attr_values | 输入 | 待设置的属性名和属性值。 |

## 返回值说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| - | graphStatus | GRAPH_SUCCESS(0)：成功。<br>GRAPH_PARAM_INVALID：属性名为空。<br>其他值：失败。 |

## 约束说明

无。

## 调用示例

```c++
std::map<AscendString, AttrValue> attr_values;
AttrValue attr_value;
attr_value.SetAttrValue(int64_t{1});
attr_values.emplace(AscendString("attr_name"), attr_value);
node.SetAttrs(attr_values);
```
