# Init（MetaMultiFunc类）

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

用户自定义flow func的初始化函数。

## 函数原型

```cpp
int32_t Init(const std::shared_ptr<MetaParams> &params)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| MetaParams | 输入 | 多func处理函数使用的参数信息。 |

## 返回值

- 0：SUCCESS。
- other：FAILED，具体请参考[UDF错误码](udf-error-code.md)。

## 异常处理

无。

## 约束说明

无。
