# ge\_options

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

配置额外的GE配置项。

## 函数原型

```python
ge_options(ge_options)
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| ge_options | dict[str, str] | 配置GE配置项。<br>其中ge.flowGraphMemMaxSize比较重要，表示所有KV cache占用的最大内存，如果设置的过大，会压缩模型的可用内存，需根据实际情况指定。 |

## 调用示例

```python
from llm_datadist import LLMConfig
ge_options = {
    "ge.flowGraphMemMaxSize": "4106127360"
}
llm_config = LLMConfig()
llm_config.ge_options = ge_options
```

## 返回值

正常情况下无返回值。

参数错误可能抛出TypeError或ValueError。

## 约束说明

无
