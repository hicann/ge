# generate\_options

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

生成配置项字典。

## 函数原型

```python
generate_options()
```

## 参数说明

无

## 调用示例

```python
from llm_datadist import LLMConfig
llm_config = LLMConfig()
...
engine_options = llm_config.generate_options()
```

## 返回值

返回配置项字典。

## 约束说明

无
