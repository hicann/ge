# AllocEmptyDataMsg（MetaContext类）

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

申请空数据的MsgType类型的message。该函数供[Proc](Proc.md)调用。

## 函数原型

```cpp
std::shared_ptr<FlowMsg> AllocEmptyDataMsg(MsgType msgType)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| msgType | 输入 | 要申请空数据的消息类型。 |

## 返回值

- 0：SUCCESS。
- other：FAILED，具体请参考[UDF错误码](udf-error-code.md)。

## 异常处理

无。

## 约束说明

无。
