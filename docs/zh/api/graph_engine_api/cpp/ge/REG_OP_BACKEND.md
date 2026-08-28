# REG\_OP\_BACKEND

## 产品支持情况

全量芯片支持。

## 头文件

\#include <graph/custom\_op.h\>

## 功能说明

开发人员可以选择将自定义算子实现类注册到指定的算子类型和后端，由框架在编译最开始调用REG\_OP\_BACKEND进行自定义算子注册。

## 函数原型

```c++
REG_OP_BACKEND(custom_op_class, op_type, backend)
```

## 参数说明

| 参数名 | 输入/输出 | 描述                                                                                        |
| --- | --- |---------------------------------------------------------------------------------------------|
| custom_op_class | 输入 | 自定义算子实现类。                |
| op_type | 输入 | 注册的算子类型名称。 |
| backend | 输入 | 自定义算子后端类型，为枚举类[OpBackend](./OpBackend.md)。                                   |

## 返回值说明

无

## 约束说明

- 同一个op_type可以分别注册不同backend的自定义算子实现，同一个backend下只能注册一个自定义算子实现。

## 调用示例

```c++
#include "graph/custom_op.h"

class AddHostCpu final : public ge::HostCpuExecuteOp {
 public:
  ge::graphStatus Execute(gert::HostCpuOpExecutionContext *ctx) override {
    // Host CPU执行逻辑
    return ge::GRAPH_SUCCESS;
  }
};

REG_OP_BACKEND(AddHostCpu, "Add", ge::OpBackend::kHostCPU);
```
