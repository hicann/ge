# aclmdlDestroyDataset

## 产品支持情况

<!-- npu="950" id197 -->
- Ascend 950PR&950DT系列产品：支持
<!-- end id197 -->
<!-- npu="A3" id198 -->
- Atlas A3系列产品：支持
<!-- end id198 -->
<!-- npu="910b" id199 -->
- Atlas A2系列产品：支持
<!-- end id199 -->
<!-- npu="310b" id200 -->
- Atlas 200I/500 A2推理产品：支持
<!-- end id200 -->
<!-- npu="310p" id201 -->
- Atlas推理系列产品：支持
<!-- end id201 -->
<!-- npu="910" id202 -->
- Atlas训练系列产品：支持
<!-- end id202 -->
<!-- npu="IPV350" id203 -->
- IPV350：支持
<!-- end id203 -->
<!-- @ref: ge/res/docs/zh/api/graph_engine_api/c/acl/aclmdlDestroyDataset_res.md#id1 -->

## 功能说明

销毁通过[aclmdlCreateDataset](aclmdlCreateDataset.md)接口创建的aclmdlDataset类型的数据。

## 函数原型

```c
aclError aclmdlDestroyDataset(const aclmdlDataset *dataset)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| dataset | 输入 | 待销毁的aclmdlDataset类型的指针。 |

## 返回值说明

返回0表示成功，返回其他值表示失败，请参见[aclError](aclError.md)。
