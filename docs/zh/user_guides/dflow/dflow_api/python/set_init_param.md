# set\_init\_param

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

设置FuncProcessPoint的初始化参数。

## 函数原型

```python
set_init_param(attr_name, attr_value)
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| attr_name | str | 初始化参数名。 |
| attr_value | Union[str,List[str],int,List[int],List[List[int]],float,List[float],bool,List[bool],DType,List[DType]] | 初始化参数值。 |

## 返回值

正常场景下返回None。

返回“TypeError”表示参数类型不正确。

## 调用示例

```python
import dataflow as df
pp = df.FuncProcessPoint(...)
pp0.set_init_param("out_type", df.DT_INT32) # 按UDF实际实现来设置
```

## 约束说明

无
