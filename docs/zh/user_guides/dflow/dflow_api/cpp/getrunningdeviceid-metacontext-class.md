# GetRunningDeviceId（MetaContext类）

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

获取正在运行的设备ID。该函数供[Init（MetaFlowFunc类）](init-metaflowfunc-class.md)调用。

## 函数原型

```cpp
virtual int32_t GetRunningDeviceId() const = 0
```

## 参数说明

无

## 返回值

返回运行的设备ID。

## 异常处理

无。

## 约束说明

无。
