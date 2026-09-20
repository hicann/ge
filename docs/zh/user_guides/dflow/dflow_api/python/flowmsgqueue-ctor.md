# FlowMsgQueue构造函数

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

FlowMsgQueue构造函数和析构函数。

## 函数原型

```python
__init__(self, flow_msg_queue: flowfunc_wrapper.FlowMsgQueue) -> None
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| flowfunc_wrapper.FlowMsgQueue | 输入 | flowfunc_wrapper模块中定义的FlowMsgQueue。实际执行时由C++代码传入，通过pybind11的绑定关系映射成flowfunc_wrapper的FlowMsgQueue对象。 |

## 返回值

返回FlowMsgQueue类型的对象。

## 异常处理

无

## 约束说明

继承Python的queue.Queue类型，但只支持get、get\_nowait、full、empty、qsize 5种接口。

流式输入（即flow func函数入参为队列）场景下，DataFlow不支持数据对齐和UDF主动上报异常。
