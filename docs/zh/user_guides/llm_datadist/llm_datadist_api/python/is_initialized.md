# is\_initialized

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

查询KvCacheManager实例是否已初始化。

## 函数原型

```python
is_initialized() -> bool
```

## 参数说明

NA

## 调用示例

```python
from llm_datadist import LLMDataDist
llm_datadist = LLMDataDist(LLMRole.Decode, 0)
...
llm_datadist.init(engine_options)
kv_cache_manager = llm_datadist.kv_cache_manager
is_initialized = kv_cache_manager.is_initialized()
```

## 返回值

如果LLMDataDist已初始化，则返回True。

如果LLMDataDist已销毁，则返回False。

## 约束说明

如果该接口返回False，则不应调用[KvCacheManager](KvCacheManager.md)的任何接口。
