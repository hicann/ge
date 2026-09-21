# deserialize\_from\_file

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

从序列化的pickle文件进行反序列化恢复Python对象。

## 函数原型

```python
deserialize_from_file(pkl_file, work_path=None)
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| pkl_file | str | 序列化的pickle文件路径。 |
| work_path | str | 序列化时的工作路径。 |

## 返回值

反序列化恢复的Python对象。

## 调用示例

```python
import dataflow as df
obj = df.msg_type_register.deserialize_from_file('file1')
```

## 约束说明

无
