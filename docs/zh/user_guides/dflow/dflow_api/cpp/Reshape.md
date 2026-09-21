# Reshape

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

对Tensor进行Reshape操作，不改变Tensor的内容。

## 函数原型

```cpp
int32_t Reshape(const std::vector<int64_t> &shape)
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| shape | 输入 | 要改变的目标shape。<br>要求shape元素个数必须和原来shape的个数一致。 |

## 返回值

- 0：SUCCESS。
- other：FAILED，具体请参考错误码。

## 异常处理

无。

## 约束说明

如果对输入进行Reshape动作，可能会影响其他使用本输入的节点正常执行。
