# addr

## 产品支持情况

全量芯片支持。

## 功能说明

获取Tensor数据在运行时的地址。

## 函数原型

```python
tensor.addr -> int
```

## 参数说明

无

## 返回值说明

返回Tensor数据地址的整数表示。

## 约束说明

- `Tensor`由运行时提供，只能在当前执行回调期间使用。
- 该地址仅用于运行时执行参数构造，不应在Python中对其进行内存读写或释放。

## 调用示例

```python
def execute(self, x: Tensor, y: Tensor) -> None:
    address = x.addr
```
