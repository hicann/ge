# aclmdlBundleCreateQueryInfo

## 产品支持情况

<!-- npu="950" id1175 -->
- Ascend 950PR&950DT系列产品：支持
<!-- end id1175 -->
<!-- npu="A3" id1176 -->
- Atlas A3系列产品：支持
<!-- end id1176 -->
<!-- npu="910b" id1177 -->
- Atlas A2系列产品：支持
<!-- end id1177 -->
<!-- npu="310b" id1178 -->
- Atlas 200I/500 A2推理产品：支持
<!-- end id1178 -->
<!-- npu="310p" id1179 -->
- Atlas推理系列产品：支持
<!-- end id1179 -->
<!-- npu="910" id1180 -->
- Atlas训练系列产品：支持
<!-- end id1180 -->
<!-- npu="IPV350" id1181 -->
- IPV350：不支持
<!-- end id1181 -->
<!-- @ref: ge/res/docs/zh/api/graph_engine_api/c/acl/aclmdlBundleCreateQueryInfo_res.md#id1 -->

## 功能说明

创建aclmdlBundleQueryInfo类型的数据，表示模型描述信息。

如需销毁aclmdlBundleQueryInfo类型的数据，请参见[aclmdlBundleDestroyQueryInfo](aclmdlBundleDestroyQueryInfo.md)。

## 函数原型

```c
aclmdlBundleQueryInfo *aclmdlBundleCreateQueryInfo()
```

## 参数说明

无

## 返回值说明

返回aclmdlBundleQueryInfo类型的指针。
