# aclmdlDestroyDesc

## 产品支持情况

<!-- npu="950" id400 -->
- Ascend 950PR&950DT系列产品：支持
<!-- end id400 -->
<!-- npu="A3" id401 -->
- Atlas A3系列产品：支持
<!-- end id401 -->
<!-- npu="910b" id402 -->
- Atlas A2系列产品：支持
<!-- end id402 -->
<!-- npu="310b" id403 -->
- Atlas 200I/500 A2推理产品：支持
<!-- end id403 -->
<!-- npu="310p" id404 -->
- Atlas推理系列产品：支持
<!-- end id404 -->
<!-- npu="910" id405 -->
- Atlas训练系列产品：支持
<!-- end id405 -->
<!-- npu="IPV350" id406 -->
- IPV350：支持
<!-- end id406 -->
<!-- @ref: ge/res/docs/zh/api/graph_engine_api/c/acl/aclmdlDestroyDesc_res.md#id1 -->

## 功能说明

销毁通过[aclmdlCreateDesc](aclmdlCreateDesc.md)接口创建的aclmdlDesc类型的数据。

## 函数原型

```c
aclError aclmdlDestroyDesc(aclmdlDesc *modelDesc)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| modelDesc | 输入 | 待销毁的aclmdlDesc类型的指针。 |

## 返回值说明

返回0表示成功，返回其他值表示失败，请参见[aclError](aclError.md)。
