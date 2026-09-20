# FunctionPp构造函数和析构函数

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

FunctionPp的构造函数和析构函数，构造函数会返回一个FunctionPp对象。

## 函数原型

```cpp
FunctionPp(const char_t *pp_name)
~FunctionPp() override
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| pp_name | 输入 | FunctionPp的名称，需要全图唯一。 |

## 返回值

返回一个FunctionPp对象。

## 异常处理

无。

## 约束说明

无。
