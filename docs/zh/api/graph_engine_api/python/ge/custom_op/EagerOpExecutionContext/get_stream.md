# get\_stream

## 产品支持情况

全量芯片支持。

## 功能说明

获取当前算子绑定的运行时执行流句柄。

## 函数原型

```python
get_stream() -> int
```

## 参数说明

无

## 返回值说明

| 类型 | 说明 |
| :--- | :--- |
| int | 当前运行时执行流句柄的整数表示。 |

## 约束说明

- 仅可在当前`execute`调用栈内调用。
- 获取失败时抛出`RuntimeError`。

## 调用示例

```python
from ge.custom_op import get_execute_ctx


def execute(self, x) -> None:
    ctx = get_execute_ctx()
    stream = ctx.get_stream()
```
