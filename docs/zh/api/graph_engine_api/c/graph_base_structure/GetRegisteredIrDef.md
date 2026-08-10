# GetRegisteredIrDef

## 产品支持情况

请参见[Session接口产品支持情况](../../cpp/ge/Session/overview.md)。

## 头文件/库文件

- 头文件：\#include <ge/ge\_api.h\>
- 库文件：libge\_runner.so

## 功能说明

获取注册的IR（Intermediate Representation）算子原型定义信息。

## 函数原型

```c
ge::Status GetRegisteredIrDef(const char *op_type, std::vector<std::pair<ge::AscendString, ge::AscendString>> &inputs, std::vector<std::pair<ge::AscendString, ge::AscendString>> &outputs, std::vector<std::pair<ge::AscendString, ge::AscendString>> &attrs)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| op_type | 输入 | 算子类型。 |
| inputs | 输入 | 算子的输入名称及类型，类型包括required/dynamic/optional。 |
| outputs | 输入 | 算子的输出名称及类型，类型包括required/dynamic。 |
| attrs | 输入 | 算子属性名称及属性类型，类型包括（与已有接口保持风格一致，数据类型前带有VT_前缀）INT/FLOAT/STRING/BOOL/DATA_TYPE/TENSOR/NAMED_ATTRS/LIST_INT/LIST_FLOAT/LIST_STRING/LIST_BOOL/LIST_TENSOR/BYTES/LIST_LIST_INT/LIST_NAMED_ATTRS。 |

## 返回值说明

| 参数名 | 类型 | 说明 |
| --- | --- | --- |
| - | Status | SUCCESS：查询成功。<br>FAILED：查询失败。 |

## 约束说明

无
