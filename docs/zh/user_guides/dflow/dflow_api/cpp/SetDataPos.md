# SetDataPos

## 产品支持情况

<!-- npu="A3" id1 -->
- Atlas A3系列产品：支持
<!-- end id1 -->
<!-- npu="910b" id2 -->
- Atlas A2系列产品：支持
<!-- end id2 -->

## 函数功能

设置输出数据对应权重矩阵中的位置。

## 函数原型

```cpp
void SetDataPos(const std::vector<std::pair<int32_t, int32_t>> &dataPos) = 0
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| --- | --- | --- |
| dataPos | 输入 | 输出数据对应的权重矩阵位置。 |

## 返回值

无。

## 异常处理

无。

## 约束说明

无。
