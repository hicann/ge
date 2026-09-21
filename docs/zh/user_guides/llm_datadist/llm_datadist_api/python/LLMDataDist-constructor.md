# LLMDataDist-constructor

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

构造LLMDataDist。

## 函数原型

```python
__init__(role: LLMRole, cluster_id: int)
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| role | [LLMRole](LLMRole.md) | 集群角色。取值如下。<br><br>  - LLMRole.DECODER：增量集群，只能作为Client使用<br>  - LLMRole.PROMPT：全量集群，只能作为Server使用<br>  - LLMRole.MIX：混合部署 |
| cluster_id | int | 集群ID。LLMDataDist标识，在所有参与建链的范围内需要确保唯一。 |

## 调用示例

```python
from llm_datadist import LLMDataDist, LLMRole
llm_datadist = LLMDataDist(LLMRole.DECODER, 0)
```

## 返回值

正常情况下返回LLMDataDist的实例。

参数错误可能抛出TypeError或ValueError。

## 约束说明

无
