# aclGetTensorDescFormat

## 产品支持情况

<!-- npu="950" id43 -->
- Ascend 950PR&950DT系列产品：支持
<!-- end id43 -->
<!-- npu="A3" id44 -->
- Atlas A3系列产品：支持
<!-- end id44 -->
<!-- npu="910b" id45 -->
- Atlas A2系列产品：支持
<!-- end id45 -->
<!-- npu="310b" id46 -->
- Atlas 200I/500 A2推理产品：支持
<!-- end id46 -->
<!-- npu="310p" id47 -->
- Atlas推理系列产品：支持
<!-- end id47 -->
<!-- npu="910" id48 -->
- Atlas训练系列产品：支持
<!-- end id48 -->
<!-- @ref: ge/res/docs/zh/api/graph_engine_api/c/acl/aclGetTensorDescFormat_res.md#id1 -->

## 功能说明

获取tensor描述中的format。

## 函数原型

```c
aclFormat aclGetTensorDescFormat(const aclTensorDesc *desc)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| desc | 输入 | aclTensorDesc类型的指针。<br>需提前调用[aclCreateTensorDesc](aclCreateTensorDesc.md)接口创建aclTensorDesc类型。 |

## 返回值说明

返回指定tensor描述的format，类型为aclFormat。
