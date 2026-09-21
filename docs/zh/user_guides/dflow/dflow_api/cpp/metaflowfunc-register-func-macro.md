# MetaFlowFunc注册函数宏

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

注册MetaFlowFunc的实现类。

## 函数原型

```cpp
REGISTER_FLOW_FUNC(name, clazz)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| Name | 输入 | 用户自定义的函数名。 |
| clazz | 输入 | MetaFlowFunc类实现的类名。 |

## 返回值

无。

## 异常处理

无。

## 约束说明

无。
