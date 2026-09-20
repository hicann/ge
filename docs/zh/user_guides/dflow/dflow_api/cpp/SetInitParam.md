# SetInitParam

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

设置FunctionPp的初始化参数，返回设置好的FunctionPp。

## 函数原型

```cpp
FunctionPp &SetInitParam(const char_t *attr_name, const ge::AscendString &value)
FunctionPp &SetInitParam(const char_t *attr_name, const char_t *value)
FunctionPp &SetInitParam(const char_t *attr_name, const std::vector<ge::AscendString> &value)
FunctionPp &SetInitParam(const char_t *attr_name, const int64_t &value)
FunctionPp &SetInitParam(const char_t *attr_name, const std::vector<int64_t> &value)
FunctionPp &SetInitParam(const char_t *attr_name, const std::vector<std::vector<int64_t>> &value)
FunctionPp &SetInitParam(const char_t *attr_name, const float &value)
FunctionPp &SetInitParam(const char_t *attr_name, const std::vector<float> &value)
FunctionPp &SetInitParam(const char_t *attr_name, const bool &value)
FunctionPp &SetInitParam(const char_t *attr_name, const std::vector<bool> &value)
FunctionPp &SetInitParam(const char_t *attr_name, const ge::DataType &value)
FunctionPp &SetInitParam(const char_t *attr_name, const std::vector<ge::DataType> &value)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| attr_name | 输入 | 初始化参数名。 |
| value | 输入 | 初始化参数值。 |

## 返回值

返回设置好的FunctionPp。

## 异常处理

无。

## 约束说明

无。
