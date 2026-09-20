# AllocTensorMsg（MetaRunContext类）

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

根据shape和data type申请Tensor类型的msg。该函数供[Proc](Proc.md)调用。

## 函数原型

```cpp
std::shared_ptr<FlowMsg> AllocTensorMsg(const std::vector<int64_t> &shape, TensorDataType dataType)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| Shape | 输入 | Tensor的Shape。 |
| dataType | 输入 | Tensor的dataType。 |

## 返回值

申请的Tensor指针。

## 异常处理

申请不到Tensor指针则返回NULL。

## 约束说明

无。
