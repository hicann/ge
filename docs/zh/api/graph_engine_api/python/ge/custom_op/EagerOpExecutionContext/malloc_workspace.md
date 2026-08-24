# malloc\_workspace

## 产品支持情况

全量芯片支持。

## 功能说明

分配Workspace内存，Placement为Device。
内存由Context构造方管理，接口调用者不需要主动释放。

## 函数原型

```python
malloc_workspace(size: int) -> int
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| :--- | :--- | :--- |
| size | 输入 | 内存大小，单位为字节。 |

## 返回值说明

| 类型 | 说明 |
| :--- | :--- |
| int | workspace的device地址。 |

## 约束说明

- 仅可在当前`execute`调用栈内调用。
- 申请失败时抛出`RuntimeError`。

## 调用示例

```python
from ge.custom_op import get_execute_ctx


def execute(self, x) -> None:
    ctx = get_execute_ctx()
    workspace = ctx.malloc_workspace(256)
```
