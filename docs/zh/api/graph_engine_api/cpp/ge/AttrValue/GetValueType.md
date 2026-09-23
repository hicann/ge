# GetValueType

## 产品支持情况

全量芯片支持。

## 头文件/库文件

- 头文件：\#include <graph/attr_value.h\>
- 库文件：libgraph.so

## 功能说明

获取 `AttrValue` 当前持有的值类型。

## 函数原型

```c++
AttrType GetValueType() const
```

## 参数说明

无。

## 返回值说明

返回 `AttrType`。支持的枚举值包括：

| 枚举值 | 对应类型 |
| --- | --- |
| `AT_NONE` | 未设置或不支持的类型 |
| `AT_INT` | `int64_t` |
| `AT_FLOAT` | `float32_t` |
| `AT_STRING` | `AscendString` |
| `AT_BOOL` | `bool` |
| `AT_TENSOR` | `Tensor` |
| `AT_DATA_TYPE` | `ge::DataType` |
| `AT_LIST_INT` | `std::vector<int64_t>` |
| `AT_LIST_FLOAT` | `std::vector<float32_t>` |
| `AT_LIST_STRING` | `std::vector<AscendString>` |
| `AT_LIST_BOOL` | `std::vector<bool>` |
| `AT_LIST_TENSOR` | `std::vector<Tensor>` |
| `AT_LIST_LIST_INT` | `std::vector<std::vector<int64_t>>` |
| `AT_LIST_DATA_TYPE` | `std::vector<ge::DataType>` |

## 约束说明

枚举值只允许在末尾追加，使用 `switch` 判断时需要包含 `default` 分支。

## 调用示例

```c++
ge::AttrValue value;
value.SetAttrValue(static_cast<int64_t>(42));
if (value.GetValueType() == ge::AttrValue::AT_INT) {
  int64_t int_value = 0;
  value.GetAttrValue(int_value);
}
```
