# GetRequiredOutputTensor

## 产品支持情况

全量芯片支持。

## 头文件/库文件

- 头文件：\#include <exe\_graph/runtime/annotated\_args\_context.h\>
- 库文件：liblowering.so

## 功能说明

基于算子IR原型定义，获取必选输出的Tensor指针。该接口将IR原型索引映射到对应输出实例，与[`GetOutputTensor`](GetOutputTensor.md)使用的扁平实例索引不同。

## 函数原型

```c++
const Tensor *GetRequiredOutputTensor(size_t ir_index) const
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| ir_index | 输入 | 该必选输出在算子IR原型定义中的0-based输出索引。 |

## 返回值说明

返回对应必选输出的只读Tensor指针；Context异常、`ir_index`越界、IR到实例的映射不存在或该输出没有实例时返回`nullptr`。

## 约束说明

无
