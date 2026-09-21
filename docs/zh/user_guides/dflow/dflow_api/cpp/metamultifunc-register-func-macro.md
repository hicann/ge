# MetaMultiFunc注册函数宏

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

注册MetaMultiFunc的实现类。

## 函数原型

```cpp
FLOW_FUNC_REGISTRAR(clazz)
```

> [!NOTE]说明
>该函数的使用示例如下：
>FLOW\_FUNC\_REGISTRAR\(UserFlowFunc\).RegProcFunc\("xxx\_func", &UserFlowFunc::Proc1\).RegProcFunc\("xxx\_func", &UserFlowFunc::Proc2\);

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| clazz | 输入 | MetaMultiFunc类实现的类名。 |

## 返回值

无。

## 异常处理

无。

## 约束说明

无。
