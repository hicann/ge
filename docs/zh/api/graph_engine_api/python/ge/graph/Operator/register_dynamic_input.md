# register\_dynamic\_input

## 产品支持情况

全量芯片支持。

## 功能说明

为算子注册动态输入端口组，声明端口组名称和当前节点的实例数量。动态输入用于输入数量不固定的算子（如`Sum`），解析器按实例数量完成连线。

## 函数原型

```python
Operator.register_dynamic_input(name: str, count: int) -> None
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| :--- | :--- | :--- |
| name | 输入 | 动态输入端口组名称，类型为非空字符串。 |
| count | 输入 | 当前节点的动态输入实例数量，类型为非负整数，取值范围为[0, 2^32-1]。 |

## 返回值说明

无。

## 约束说明

- 仅在回调执行期间可用，回调结束后调用会抛出`RuntimeError`。
- source算子为只读对象，调用会抛出`RuntimeError`。
- 动态端口注册的是当前节点需要的实例数量，不改变算子原型的端口定义。
- `name`不是非空字符串时，抛出`TypeError`。
- `count`不是整数时，抛出`TypeError`；`count`为负数或超出取值范围时，抛出`ValueError`。

## 调用示例

```python
target.register_dynamic_input("x", count)
```

按ONNX节点的输入数量注册动态输入并记录数量属性的完整示例，参见[`parse_node`](../../onnx_plugin/OnnxPlugin/parse_node.md)。
