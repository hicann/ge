# 简介

用于执行算子的上下文环境。

为算子在host上运行时提供输入输出管理、内存分配等运行时支持。

## 需要包含的头文件

```c++
#include <exe_graph/runtime/host_cpu_op_execution_context.h>
```

## Public成员函数

```c++
const Tensor *GetInputTensor(size_t index) const
const Tensor *GetRequiredInputTensor(size_t ir_index) const
const Tensor *GetOptionalInputTensor(size_t ir_index) const
const Tensor *GetDynamicInputTensor(size_t ir_index, size_t relative_index) const
const Tensor *GetOutputTensor(size_t index) const
Tensor *MallocOutputTensor(size_t index, const StorageShape &shape, const StorageFormat &format, ge::DataType dtype)
Tensor *MakeOutputRefInput(size_t output_index, size_t input_index)
```
