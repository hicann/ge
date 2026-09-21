# kv\_cache\_manager

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

获取KvCacheManager实例。

## 函数原型

```python
kv_cache_manager()
```

## 参数说明

无

## 调用示例

```python
from llm_datadist import LLMDataDist, LLMRole
llm_datadist = LLMDataDist(LLMRole.DECODER, 0)
...
llm_datadist.init(engine_options)
kv_cache_manager = llm_datadist.kv_cache_manager
```

## 返回值

返回KvCacheManager实例。

## 约束说明

无
