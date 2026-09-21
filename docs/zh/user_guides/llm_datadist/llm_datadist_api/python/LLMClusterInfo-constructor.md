# LLMClusterInfo-constructor

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

构造LLMClusterInfo，用于[link\_clusters](link_clusters.md)和[unlink\_clusters](unlink_clusters.md)接口的参数类型。

## 函数原型

```python
__init__()
```

## 参数说明

无

## 调用示例

```python
from llm_datadist import LLMClusterInfo
llm_cluster = LLMClusterInfo()
```

## 返回值

返回LLMClusterInfo的实例。

## 约束说明

无
