# SetUserData（DataFlowInfo数据类型）

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

设置用户信息。

## 函数原型

```cpp
Status SetUserData(const void *data, size_t size, size_t offset = 0U)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| data | 输入 | 用户数据指针。 |
| size | 输入 | 用户数据长度。取值范围(0, 64]。 |
| offset | 输入 | 用户数据的偏移值，需要遵循如下约束。<br>[0, 64), size + offset <= 64 |

## 返回值

- 0：SUCCESS。
- other：FAILED。

## 异常处理

无。

## 约束说明

无。
