# GetLogger

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

获取日志实现类。

## 函数原型

```cpp
static FlowFuncLogger &GetLogger(FlowFuncLogType type)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| type | 输入 | 日志类型取值为FlowFuncLogType<br>enum class FLOW_FUNC_VISIBILITY FlowFuncLogType {<br>DEBUG_LOG = 0, // 调试日志<br>RUN_LOG = 1  // 运行日志<br>}; |

## 返回值

日志实现类。

## 异常处理

无。

## 约束说明

无。
