# aclGetTensorDescAddress

## 产品支持情况

<!-- npu="950" id1126 -->
- Ascend 950PR&950DT系列产品：支持
<!-- end id1126 -->
<!-- npu="A3" id1127 -->
- Atlas A3系列产品：支持
<!-- end id1127 -->
<!-- npu="910b" id1128 -->
- Atlas A2系列产品：支持
<!-- end id1128 -->
<!-- npu="310b" id1129 -->
- Atlas 200I/500 A2推理产品：支持
<!-- end id1129 -->
<!-- npu="310p" id1130 -->
- Atlas推理系列产品：支持
<!-- end id1130 -->
<!-- npu="910" id1131 -->
- Atlas训练系列产品：支持
<!-- end id1131 -->
<!-- @ref: ge/res/docs/zh/api/graph_engine_api/c/acl/aclGetTensorDescAddress_res.md#id1 -->

## 功能说明

获取指定算子输入/输出的tensor数据的内存地址。

## 函数原型

```c
void *aclGetTensorDescAddress(const aclTensorDesc *desc)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| desc | 输入 | aclTensorDesc类型的指针。<br>调用[aclGetTensorDescByIndex](aclGetTensorDescByIndex.md)接口获取算子的指定输入/输出的tensor描述，作为本接口的输入。 |

## 返回值说明

返回指定算子输入/输出的tensor数据的内存地址。
