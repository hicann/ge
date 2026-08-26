# get\_platform\_resource\_group

## 产品支持情况

全量芯片支持。

## 功能说明

查询一个平台资源组。

## 函数原型

```python
get_platform_resource_group(group: str) -> dict[str, str]
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| :--- | :--- | :--- |
| group | 输入 | 非空平台资源组名称。 |

## 约束说明

只能在当前同步`compile`回调内调用。

## 调用示例

```python
resources = platform.get_platform_resource_group("ai_core")
```
