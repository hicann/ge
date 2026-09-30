# GetRequiredOutputTensor

## 产品支持情况

全量芯片支持。

## 头文件/库文件

- 头文件：\#include <exe\_graph/runtime/resource\_usage\_context.h\>
- 库文件：liblowering.so

## 功能说明

基于算子IR原型定义，获取REQUIRED\_OUTPUT（必选输出）类型的输出Tensor指针。该接口将IR原型索引映射到对应输出实例，与[`GetOutputTensor`](GetOutputTensor.md)使用的扁平实例索引不同，可读取编译期静态shape、format与dtype。

## 函数原型

```c++
const Tensor *GetRequiredOutputTensor(size_t ir_index) const
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| ir_index | 输入 | IR原型定义中的索引。 |

## 返回值说明

Tensor指针，异常或该输出未实例化时返回空指针。

## 约束说明

无
