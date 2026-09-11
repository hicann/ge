# aclmdlQueryWorkspaceSize

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：支持
<!-- end id3 -->
<!-- npu="310b" id4 -->
- Atlas 200I/500 A2 推理产品：支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas 推理系列产品：支持
<!-- end id5 -->
<!-- npu="910" id6 -->
- Atlas 训练系列产品：支持
<!-- end id6 -->
<!-- npu="IPV350" id7 -->
- IPV350：不支持
<!-- end id7 -->

## 头文件/库文件

```c
#include "acl/acl_mdl.h"
```

库文件：`libacl_mdl.so`。

## 功能说明

根据模型文件和工作内存优化模式，获取模型执行时所需的工作内存大小。

当由用户管理内存时，为确保内存不浪费，在申请工作内存前，需要调用本接口查询模型运行时所需工作内存的大小。相比[aclmdlQuerySize](aclmdlQuerySize.md)，本接口可通过`memOptimizeMode`指定是否查询输入输出内存优化后的工作内存大小。

如果模型输入数据的Shape不确定，则不能调用本接口查询内存大小，在加载模型时，就无法由用户管理内存，因此需选择由系统管理内存的模型加载接口（例如，[aclmdlLoadFromFile](aclmdlLoadFromFile.md)、[aclmdlLoadFromMem](aclmdlLoadFromMem.md)）。

## 函数原型

```c
aclError aclmdlQueryWorkspaceSize(const char *fileName, size_t memOptimizeMode, size_t *workSize)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| fileName | 输入 | 模型文件路径的指针，路径中包含文件名。运行程序（APP）的用户需要对该路径有访问权限。<br>此处的模型文件是OM2模型文件。<br>关于如何获取OM2文件，请参见《[ATC离线模型编译工具](../../../../user_guides/atc_tools/README.md)》中的“[--mode](../../../../user_guides/atc_tools/CLI_options/--mode.md)”。 |
| memOptimizeMode | 输入 | 工作内存优化模式。<br>取值范围：`ACL_WORKSPACE_MEM_OPTIMIZE_DEFAULT`或`ACL_WORKSPACE_MEM_OPTIMIZE_INPUTOUTPUT`。<br>`ACL_WORKSPACE_MEM_OPTIMIZE_DEFAULT`表示查询模型执行所需的完整工作内存大小。<br>`ACL_WORKSPACE_MEM_OPTIMIZE_INPUTOUTPUT`表示查询输入输出内存优化后的工作内存大小。 |
| workSize | 输出 | 模型执行时所需的工作内存大小的指针，单位Byte。<br>此处的内存为Device内存，而且需要用户申请和释放。 |

## 返回值说明

返回0表示成功，返回其他值表示失败，请参见[aclError](aclError.md)。

## 约束说明

- 本接口仅支持OM2模型文件。OM模型文件调用本接口时返回不支持。
- 当`memOptimizeMode`设置为`ACL_WORKSPACE_MEM_OPTIMIZE_INPUTOUTPUT`时，如果模型文件不支持输入输出内存优化，则`workSize`返回模型执行所需的完整工作内存大小。
