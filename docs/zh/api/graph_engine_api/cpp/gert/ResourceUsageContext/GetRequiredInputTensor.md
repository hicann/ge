# GetRequiredInputTensor

## 产品支持情况

全量芯片支持。

## 头文件/库文件

- 头文件：\#include <exe\_graph/runtime/resource\_usage\_context.h\>
- 库文件：liblowering.so

## 功能说明

基于算子IR原型定义，获取REQUIRED\_INPUT（必需输入）类型的输入Tensor指针，使用IR原型定义中的索引定位，可读取编译期静态shape、format与dtype。

## 函数原型

```c++
const Tensor *GetRequiredInputTensor(size_t ir_index) const
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| ir_index | 输入 | IR原型定义中的索引。 |

## 返回值说明

Tensor指针，异常或该输入未实例化时返回空指针。

## 约束说明

- 仅在`DeclareResourceUsage`回调期间有效，不得缓存到回调之外使用。
