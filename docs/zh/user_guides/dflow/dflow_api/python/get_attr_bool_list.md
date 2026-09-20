# get\_attr\_bool\_list

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

获取指定名称的bool数组类型属性值。

## 函数原型

```python
get_attr_bool_list(self, name: str) -> Tuple[int, List[bool]]
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| name | 输入 | 属性名。 |

## 返回值

获取返回码及bool数组类型的属性值。

- 如果该属性存在，返回的Tuple中第一个元素为FLOW\_FUNC\_SUCCESS，第二个元素为bool数组的list。
- 如果属性不存在，Tuple中仅包含错误码一个元素。

## 异常处理

无

## 约束说明

无
