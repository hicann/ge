# aclRecoverAllHcclTasks

**须知：本接口为试验特性，后续版本可能会存在变更，不支持应用于生产环境中。**

### 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- Atlas 200I/500 A2 推理产品：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas 推理系列产品：不支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- Atlas 训练系列产品：不支持
<!-- end id6 -->
<!-- npu="IPV350" id7 -->
- IPV350：不支持
<!-- end id7 -->
<!-- @ref: ge/res/docs/zh/api/graph_engine_api/c/acl/aclRecoverAllHcclTasks_res.md#id1 -->

## 功能说明

维测场景下，本接口恢复指定Device上的所有集合通信任务。

## 函数原型

```c
aclError aclRecoverAllHcclTasks(int32_t deviceId)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| deviceId | 输入 | Device ID。<br>用户调用aclrtGetDeviceCount接口获取可用的Device数量后，这个Device ID的取值范围：[0, (可用的Device数量-1)] |

## 返回值说明

返回0表示成功，返回其他值表示失败，请参见[aclError](aclError.md)。
