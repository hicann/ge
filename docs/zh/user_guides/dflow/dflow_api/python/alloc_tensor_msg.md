# alloc\_tensor\_msg

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

根据shape、data type以及对齐大小申请Tensor类型的FlowMsg。

## 函数原型

```python
alloc_tensor_msg(self, shape: Union[List[int], Tuple[int]], dtype, align:Optional[int] = 64) -> FlowMsg
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| shape | 输入 | Tensor的shape。 |
| dtype | 输入 | Tensor的data type。 |
| align | 输入 | 申请内存地址对齐大小，取值范围【32、64、128、256、512、1024】，默认值为64。 |

## 返回值

返回FlowMsg的实例。

## 异常处理

申请不到Tensor指针则返回None。

## 约束说明

无
