# SetMultiOutputs

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

批量设置指定index和options的输出，该函数供func函数调用。

## 函数原型

```cpp
int32_t SetMultiOutputs(uint32_t outIdx, const std::vector<std::shared_ptr<FlowMsg>> &outMsgs, const OutOptions &options) = 0
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| outIdx | 输入 | 输出index，从0开始。 |
| outMsgs | 输入 | 输出信息列表。 |
| options | 输入 | 输出对应的OutOptions。 |

## 返回值

- 0：SUCCESS。
- other：FAILED，具体请参考[UDF错误码](udf-error-code.md)。

## 异常处理

无。

## 约束说明

无。
