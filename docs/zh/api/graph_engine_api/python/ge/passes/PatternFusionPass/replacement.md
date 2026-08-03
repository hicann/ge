# replacement

## 产品支持情况

全量芯片支持。

## 功能说明

生成替换子图。

## 函数原型

```python
replacement(self, match_result: MatchResult) -> Graph
replacement(self, match_result: MatchResult, context: PassContext) -> Graph
```

表达式pattern也可使用以下写法：

```python
# 简单替换，不需要匹配详情
def replacement(self, inputs) -> TensorHolder: ...
# 需要读取捕获Tensor或节点属性
def replacement(self, inputs, match_result) -> TensorHolder: ...
# 需要读取编译期上下文
def replacement(self, inputs, context: PassContext) -> TensorHolder: ...
# 需要同时读取匹配详情和编译期上下文
def replacement(self, inputs, match_result, context: PassContext) -> TensorHolder: ...
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| self | 输入 | PatternFusionPass子类的Pass实例对象。用于访问Pass配置、调用方法。 |
| inputs | 输入 | PatternInputs类型的替换图输入集合，用于创建替换图的输入Tensor。 |
| match_result | 输入 | MatchResult类型的Pattern匹配结果，包含匹配到的节点和边信息。|
| context | 输入 | 当前编译期PassContext，仅在当前回调栈内有效，可读取编译选项或设置错误信息。 |

## 返回值说明

| 类型 | 说明 |
| --- | --- |
| Graph | 返回替换后的子图，类型为ge.graph.Graph。 |

## 约束说明

表达式replacement\(self, inputs\)可返回TensorHolder或非空TensorHolder列表/元组，Python层会自动构造替换图。需要读取匹配详情时可增加match\_result参数；只需要编译期上下文时，可以使用replacement\(self, inputs, context\)，其中第三个参数必须命名为`context`；需要同时读取匹配详情和编译期上下文时，可以使用replacement\(self, inputs, match\_result, context\)，其中第四个参数必须命名为`context`。未声明`context`时保持原有调用方式。
