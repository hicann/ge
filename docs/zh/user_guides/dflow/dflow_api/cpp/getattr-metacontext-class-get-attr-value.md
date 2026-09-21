# GetAttr（MetaContext类，获取属性值）

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

根据属性名获取对应的属性值。该函数供[Init（MetaFlowFunc类）](init-metaflowfunc-class.md)调用。

## 函数原型

```cpp
int32_t GetAttr(const char *attrName, T &value) const
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| attrName | 输入 | 属性名。 |
| value | 输出 | 属性值。 |

## 返回值

- 0：SUCCESS。
- other：FAILED，具体请参考[UDF错误码](udf-error-code.md)。

## 异常处理

无。

## 约束说明

无。
