# index

## 产品支持情况

全量芯片支持。

## 头文件/库文件

- 头文件：无
- 库文件：ge_custom_op_native.so、libge_python_custom_op_bridge.so

## 功能说明

获取当前workspace的实例索引。

## 函数原型

```python
@property
def index(self) -> int
```

## 参数说明

无

## 返回值说明

| 类型 | 说明 |
| :--- | :--- |
| int | 当前workspace的实例索引。 |

## 约束说明

- 此属性为只读属性，应使用`workspace.index`访问，不能使用`workspace.index()`调用或为其赋值。
- `WorkspaceAddr`只能由当前`declare_launch_args`回调中的`AnnotatedArgsContext.malloc_workspace()`返回，不能直接构造。
- 当前回调结束后，该对象随`AnnotatedArgsContext`失效；访问此属性时抛出`RuntimeError`。
