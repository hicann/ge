# dataflow.Framework

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

设置原始网络模型的框架类型。

## 函数原型

不涉及

## 参数说明

枚举值如下：

- Framework.TENSORFLOW
- Framework.ONNX
- Framework.MINDSPORE

## 返回值

无

## 调用示例

```python
import dataflow as df
framework = df.Framework.TENSORFLOW
pp1 = df.GraphProcessPoint(framework,...)
```

## 约束说明

无
