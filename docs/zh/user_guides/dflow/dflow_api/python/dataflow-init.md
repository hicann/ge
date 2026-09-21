# dataflow.init

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

初始化dataflow时的options。

## 函数原型

```python
init(flow_options)
```

## 参数说明

| 参数名称 | 数据类型 | 取值说明 |
| --- | --- | --- |
| flow_options | Dict[str, str] | 初始化dataflow时的options。options当前支持全局和session级别的。<br>其中的配置示例请按照Python语言进行适配。 |

## 返回值

正常场景下返回None。

异常情况下会抛出DfException异常。可以通过捕捉异常获取DfException中的error\_code与message查看具体的错误码及错误信息。详细信息请参考[DataFlow错误码](dataflow-error-code.md)。

## 调用示例

```python
import dataflow as df
df.init(...)
```

## 约束说明

无
