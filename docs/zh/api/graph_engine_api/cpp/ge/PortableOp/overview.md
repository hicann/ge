# 简介

自定义算子的数据序列化/反序列化能力接口。实现该接口后，GE会在模型保存阶段回调Serialize接口，将自定义算子数据（如kernel bin）序列化到buffer中并随离线模型（OM）保存；在模型加载阶段回调Deserialize接口，从buffer中恢复算子数据。buffer格式由用户自定义，GE不解析只透传。

未实现该接口的自定义算子同样可用于生成离线模型：实现了[AnnotatedArgsOp](../AnnotatedArgsOp/overview.md)的算子可在编译期通过DeclareLaunchArgs将kernel数据提交给GE，随模型保存并在模型加载执行阶段用于launch；否则其kernel二进制等数据不会随OM保存和恢复，需自行管理。

## 需要包含的头文件

```c++
#include <graph/custom_op.h>
```

## Public成员函数

```c++
virtual graphStatus Serialize(std::vector<uint8_t> &buffer) = 0
virtual graphStatus Deserialize(const std::vector<uint8_t> &buffer) = 0
```
