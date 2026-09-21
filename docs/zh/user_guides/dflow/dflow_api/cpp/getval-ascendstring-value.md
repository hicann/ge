# GetVal\(AscendString &value\)

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

获取string类型的属性值。

## 函数原型

```cpp
int32_t GetVal(AscendString &value) const = 0
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| value | 输出 | 获取的AscendString类型属性值。 |

## 返回值

- 0：SUCCESS。
- 其他返回值请参考[UDF错误码](udf-error-code.md)。

## 异常处理

无。

## 约束说明

无。
