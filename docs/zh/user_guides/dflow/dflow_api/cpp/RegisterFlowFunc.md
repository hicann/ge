# RegisterFlowFunc

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

注册flow func。

不建议直接使用该函数，建议使用[MetaFlowFunc注册函数宏](metaflowfunc-register-func-macro.md)来注册flow func。

## 函数原型

```cpp
FLOW_FUNC_VISIBILITY bool RegisterFlowFunc(const char *flowFuncName, const FLOW_FUNC_CREATOR_FUNC &func) noexcept
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| flowFuncName | 输入 | flow func的名称。不可以设置为NULL，必须以“\0”结尾。 |
| func | 输入 | flow func的创建函数。 |

## 返回值

- true
- false

## 异常处理

无。

## 约束说明

无。
