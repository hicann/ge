# device\_id

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

设置当前进程Device ID，对应底层ge.exec.deviceId配置项。

## 函数原型

```python
device_id(device_id)
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| device_id | Union[int, List[int], Tuple[int]] | 设置当前进程的Device ID。支持配置为一个或者列表，配置为列表时以半角逗号间隔。<br><br>  - 单进程单卡场景下，需要配置为一个，例如：0<br>  - 单进程多卡场景下，需要配置为列表，例如：[0, 1] |

## 调用示例

```python
from llm_datadist import LLMConfig
llm_config = LLMConfig()
# 单进程单卡设置方法
llm_config.device_id = 0
# 单进程多卡设置方法
# llm_config.device_id = [0, 1]
```

## 返回值

正常情况下无返回值。

参数错误可能抛出TypeError或ValueError。

## 约束说明

无
