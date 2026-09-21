# GetUserData（MetaRunContext类）

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

获取用户数据。该函数供[Proc](Proc.md)调用。

## 函数原型

```cpp
int32_t GetUserData(void *data, size_t size, size_t offset = 0U) const
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| data | 输入/输出 | 用户数据指针。 |
| size | 输入 | 用户数据长度。取值范围(0, 64]。 |
| offset | 输入 | 用户数据的偏移值，需要遵循如下约束。<br>[0, 64), size + offset <= 64 |

## 返回值

- 0：SUCCESS。
- other：FAILED，具体请参考[UDF错误码](udf-error-code.md)。

## 异常处理

无。

## 约束说明

无。
