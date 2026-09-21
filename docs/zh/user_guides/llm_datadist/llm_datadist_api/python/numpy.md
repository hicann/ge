# numpy

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2推理系列产品：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2训练系列产品：不支持
<!-- end id3 -->

## 函数功能

获取tensor的numpy数据。

## 函数原型

```python
numpy(copy=False)
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| copy | bool | 取值为False或者True。<br>默认值为False，表示从tensor转换到numpy.ndarray，且数据不做拷贝，如果取值为True，则表示需要对数据进行拷贝。<br>如果是tensor是string类型的数据，该参数需要用户设置成True，否则会抛出异常。 |

## 调用示例

```python
tensor = Tensor(numpy.array([1]))
np_arr = tensor.numpy()
```

## 返回值

返回numpy.array。

## 约束说明

无
