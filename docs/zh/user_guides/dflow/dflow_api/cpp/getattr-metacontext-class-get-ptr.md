# GetAttr（MetaContext类，获取指针）

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

根据属性名获取AttrValue类型的指针。该函数供[Init（MetaFlowFunc类）](init-metaflowfunc-class.md)调用。

## 函数原型

```cpp
std::shared_ptr<const AttrValue> GetAttr(const char  *attrName) const
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| attrname | 输入 | 属性名。 |

## 返回值

获取到的AttrValue类型的指针。

## 异常处理

无。

## 约束说明

无。
