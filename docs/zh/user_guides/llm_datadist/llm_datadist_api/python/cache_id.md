# cache\_id

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

获取KvCache的id。

## 函数原型

```python
@property
cache_id() -> int
```

## 参数说明

无

## 调用示例

```python
...
kv_cache = kv_cache_manager.allocate_cache(cache_desc, cache_keys)
print(kv_cache.cache_id)
```

## 返回值

正常情况返回类型为KvCache的id。

## 约束说明

无
