# get\_soc\_version

## 产品支持情况

全量芯片支持。

## 功能说明

查询当前编译平台的SoC版本。

## 函数原型

```python
get_soc_version() -> str
```

## 参数说明

无

## 约束说明

只能在当前同步`compile`回调内调用。

## 调用示例

```python
soc_version = platform.get_soc_version()
```
