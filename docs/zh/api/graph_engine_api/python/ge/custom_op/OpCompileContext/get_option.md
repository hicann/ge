# get\_option

## 产品支持情况

全量芯片支持。

## 功能说明

查询当前图编译上下文中的option。

## 函数原型

```python
get_option(option_key: str) -> str
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| :--- | :--- | :--- |
| option_key | 输入 | 非空option名称。 |

## 约束说明

只能在当前同步`compile`回调内调用。

## 调用示例

```python
option = ctx.get_option("custom.compile.option")
```
