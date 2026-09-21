# get\_nowait

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

无等待地获取队列中的元素，功能等同于get\(block=False\)。

## 函数原型

```python
get_nowait(self)
```

## 参数说明

无

## 返回值

MsgType中所对应类型的数据对象。

## 异常处理

队列为空时会抛出queue.Empty异常。

## 约束说明

无
