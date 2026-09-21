# append\_local\_ip\_info

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

添加本地集群IP信息。

## 函数原型

```python
append_local_ip_info(self, ip: Union[str, int], port: int)
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| ip | Union[str, int] | 本地集群Device卡IP。 |
| port | int | 本地集群Device卡端口。 |

## 调用示例

```python
llm_cluster = LLMClusterInfo()
llm_cluster.append_local_ip_info("1.1.1.1", 10000)
```

## 返回值

正常情况下无返回值。

参数错误可能抛出TypeError或ValueError。

## 约束说明

无
