# GetStreamId

## 产品支持情况

全量芯片支持。

## 头文件/库文件

- 头文件：\#include <exe\_graph/runtime/annotated\_args\_context.h\>
- 库文件：liblowering.so

## 功能说明

图框架在编译时有[多Stream并发优化技术](https://www.hiascend.com/developer/techArticles/20240701-1?envFlag=1)，该接口用于获取当前Context所属节点的逻辑StreamId。

## 函数原型

```c++
uint32_t GetStreamId() const
```

## 参数说明

无

## 返回值说明

正常时返回当前节点的逻辑主stream ID；异常时返回UINT32_MAX。

## 约束说明

无
