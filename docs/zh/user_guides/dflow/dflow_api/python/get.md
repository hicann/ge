# get

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

获取队列中的元素。

## 函数原型

```python
get(self, block=True, timeout=None)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| block | 输入 | 是否阻塞当前线程，默认值为True，表示会阻塞当前线程直到超时时间或者有元素取出；如果设置为False，队列为空时会抛出queue.Empty异常。 |
| timeout | 输入 | 队列出队的阻塞时间，单位为ms，取值范围[0, 2147483647)，默认值为None，表示会一直阻塞当前线程直到有元素取出。 |

## 返回值

MsgType中所对应类型的数据对象。

## 异常处理

如果block为False或者get超时，会抛出queue.Empty异常；

如果队列出队过程中发生重部署或者进程退出，会抛出DfAbortException异常，表示函数执行中断；

如果get失败，会抛出DfException异常，表示UDF内部出队过程中发生异常。

## 约束说明

无。
