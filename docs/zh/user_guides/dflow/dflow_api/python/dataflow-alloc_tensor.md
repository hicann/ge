# dataflow.alloc\_tensor

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

根据shape、data type以及对齐大小申请dataflow tensor。

## 函数原型

```python
alloc_tensor(shape: Union[List[int], Tuple[int]], dtype, align:Optional[int] = 64) -> Tensor
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| shape | 输入 | Tensor的shape。 |
| dtype | 输入 | Tensor的dataType。 |
| align | 输入 | 申请内存地址对齐大小，取值范围【32、64、128、256、512、1024】，默认值为64。 |

## 返回值

返回Tensor的实例。

## 异常处理

申请不到tensor指针则返回None。

## 约束说明

无
