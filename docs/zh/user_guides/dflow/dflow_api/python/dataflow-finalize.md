# dataflow.finalize

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

释放dataflow初始化的资源。

## 函数原型

```python
finalize()
```

## 参数说明

无

## 返回值

无

## 调用示例

```python
import dataflow as df
# 初始化
df.init(...)
# dataflow处理逻辑
# 释放资源
df.finalize()
```

## 约束说明

无
