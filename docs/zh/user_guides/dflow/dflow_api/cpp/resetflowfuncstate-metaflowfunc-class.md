# ResetFlowFuncState（MetaFlowFunc类）

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

在故障恢复场景下，快速重置FlowFunc为初始化状态。

## 函数原型

```cpp
int32_t ResetFlowFuncState()
```

## 参数说明

无

## 返回值

- 0：SUCCESS。
- other：FAILED，具体请参考[UDF错误码](udf-error-code.md)。

## 异常处理

无。

## 约束说明

此接口为虚函数，当用户的FlowFunc未实现ResetFlowFuncState函数时，框架默认返回FLOW\_FUNC\_ERR\_NOT\_SUPPORT（认为当前FlowFunc不支持此操作），把原来创建的FlowFunc删除掉，重新创建一个新的FlowFunc，然后再调用[Init](init-metaflowfunc-class.md)接口进行初始化。
