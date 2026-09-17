# register\_optional\_input

## 产品支持情况

全量芯片支持。

## 功能说明

为算子注册可选输入端口，用于在解析期补充算子的可选输入端口定义。

## 函数原型

```python
Operator.register_optional_input(name: str) -> None
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| :--- | :--- | :--- |
| name | 输入 | 可选输入端口名称，类型为非空字符串。 |

## 返回值说明

无。

## 约束说明

- 仅在回调执行期间可用，回调结束后调用会抛出`RuntimeError`。
- source算子为只读对象，调用会抛出`RuntimeError`。
- 目标算子原型已声明的端口不应重复注册。
- 动态输入输出原型（如`PartitionedCall`）的可选输入实例端口必须通过本方法注册，否则解析器连线阶段失败。
- `name`不是非空字符串时，抛出`TypeError`。

## 调用示例

```python
target.register_optional_input("bias")
```
