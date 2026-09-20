# mem\_utilization

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2推理系列产品：支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2训练系列产品：不支持
<!-- end id3 -->

## 函数功能

配置ge.flowGraphMemMaxSize内存的利用率。默认值0.95。

## 函数原型

```python
mem_utilization(mem_utilization)
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| mem_utilization | float | 内存利用率。默认值0.95。取值范围0.0~1.0。 |

## 调用示例

```python
from llm_datadist import LLMConfig
llm_config = LLMConfig()
llm_config.mem_utilization = 0.95
```

## 返回值

正常情况下无返回值。

参数错误可能抛出TypeError或ValueError。

## 约束说明

无。
