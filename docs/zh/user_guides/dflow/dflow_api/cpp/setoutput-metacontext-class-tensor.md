# SetOutput（MetaContext类,tensor）

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

设置指定index的output的tensor。该函数供[Proc](Proc.md)调用。

## 函数原型

```cpp
int32_t SetOutput(uint32_t outIdx, std::shared_ptr<FlowMsg> outMsg)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| outIdx | 输入 | 指定输出的index。 |
| outMsg | 输入 | 指定输出的Msg。 |

## 返回值

- 0：SUCCESS。
- other：FAILED，具体请参考[UDF错误码](udf-error-code.md)。

## 异常处理

无。

## 约束说明

无。
