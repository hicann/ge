# get\_ai\_core\_num

## 产品支持情况

全量芯片支持。

## 功能说明

查询当前编译平台的AI Core数量。

## 函数原型

```python
get_ai_core_num() -> int
```

## 参数说明

无

## 约束说明

只能在当前同步`compile`回调内调用。

## 调用示例

```python
core_num = platform.get_ai_core_num()
```
