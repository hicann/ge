# SetExceptionCatch

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

设置用户异常捕获功能是否开启。

## 函数原型

```cpp
FlowGraph &SetExceptionCatch(bool enable_exception_catch)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| enable_exception_catch | 输入 | 是否启用用户异常捕获功能。取值如下：<br><br>  - true：是，开启异常功能。<br>  - false：否，关闭异常功能。<br><br>默认值：false。 |

## 返回值

返回当前对象。

## 异常处理

无。

## 约束说明

开启异常功能时必须同时使用[SetInputsAlignAttrs](SetInputsAlignAttrs.md)接口开启数据对齐功能，否则编译报错。
