# get\_core\_num

## 产品支持情况

全量芯片支持。

## 功能说明

查询平台核数。

## 函数原型

```python
get_core_num(core_type: str | None = None) -> int
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| :--- | :--- | :--- |
| core_type | 输入 | 可选的非空核类型名称。为`None`时查询平台默认核数。 |

## 约束说明

只能在当前同步`compile`回调内调用。

## 调用示例

```python
core_num = platform.get_core_num()
```
