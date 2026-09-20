# RunFlowModel（MetaContext类）

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

同步执行指定的模型。该函数供[Proc](Proc.md)调用。

## 函数原型

```cpp
int32_t RunFlowModel(const char *modelKey, const std::vector<std::shared_ptr<FlowMsg>> &inputMsgs,
std::vector<std::shared_ptr<FlowMsg>> &outputMsgs, int32_t timeout)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| modelKey | 输入 | 指定的模型key，与AddInvokedClosure中指定的name一致。 |
| inputMsgs | 输入 | 提供给模型的输入。 |
| outputMsgs | 输出 | 模型执行的输出结果。 |
| timeout | 输入 | 等待模型执行超时时间，单位ms，-1表示永不超时。 |

## 返回值

- 0：SUCCESS。
- other：FAILED，具体请参考[UDF错误码](udf-error-code.md)。

## 异常处理

无。

## 约束说明

无。
