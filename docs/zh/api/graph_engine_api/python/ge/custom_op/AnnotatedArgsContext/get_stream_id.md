# get\_stream\_id

## 产品支持情况

全量芯片支持。

## 头文件/库文件

- 头文件：无
- 库文件：ge_custom_op_native.so、libge_python_custom_op_bridge.so

## 功能说明

获取当前`AnnotatedArgsContext`的stream标识。创建`AnnotatedKernelLaunchInfo`时可使用该标识设置`stream_id`。

## 函数原型

```python
get_stream_id() -> int
```

## 参数说明

无

## 返回值说明

| 类型 | 说明 |
| :--- | :--- |
| int | 当前`AnnotatedArgsContext`的stream标识。 |

## 约束说明

- 此方法只能在当前`declare_launch_args`回调内调用。
- 当前`AnnotatedArgsContext`已过期时，抛出`RuntimeError`。
