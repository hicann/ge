# addr

## 产品支持情况

全量芯片支持。

## 功能说明

获取当前编译回调中workspace的地址整数。

## 函数原型

```python
@property
def addr(self) -> int
```

## 参数说明

无

## 返回值说明

| 类型 | 说明 |
| :--- | :--- |
| int | 当前编译回调中workspace的地址整数。 |

## 调用示例

```python
from ge.custom_op import get_declare_launch_args_ctx


def declare_launch_args(self, x1, x2, y) -> None:
    ctx = get_declare_launch_args_ctx()
    workspace = ctx.malloc_workspace(256)
    workspace_addr = workspace.addr
```

## 约束说明

- 此属性为只读属性，应使用`workspace.addr`访问，不能使用`workspace.addr()`调用或为其赋值。
- `WorkspaceAddr`只能由当前`declare_launch_args`回调中的`AnnotatedArgsContext.malloc_workspace()`返回，不能直接构造。
- Python代码不得解引用该地址。
- 当前回调结束后，该对象随`AnnotatedArgsContext`失效；访问此属性时抛出`RuntimeError`。
