# set\_transaction\_id

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

设置DataFlow数据传输使用的事务ID。

## 函数原型

```python
set_transaction_id(self, transaction_id) -> None
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| transaction_id | 输入 | 数据传输使用的事务ID。<br>设置为0时表示不使用自定义的transaction_id，内部会采用自增的方式自动生成transaction_id。<br>在数据对齐场景下，会使用transaction_id进行路由和对齐。 |

## 返回值

无

## 异常处理

无

## 约束说明

无
