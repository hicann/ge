# can\_fuse

## 产品支持情况

全量芯片支持。

## 功能说明

在修改计算图前，检查一组主图节点是否满足融合约束，不修改计算图。检查内容包括：

- 节点集合非空，且每个Node及其底层句柄有效。
- 所有节点都具有有效的owner graph，并且属于同一张图。
- 节点中已设置的用户stream label不能出现多个不同值。
- 节点中已设置的SuperKernel scope和SuperKernel options分别不能出现多个不同值。
- 节点中已设置的AI Core数量配置不能出现多个不同值，Vector Core数量配置也不能出现多个不同值。
- 将节点集合融合成一个节点后，数据边和控制边不会形成环。

## 函数原型

```python
can_fuse(nodes: Iterable[Node]) -> FuseCheckResult
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| nodes | 输入 | 待融合的连通节点集合。元素必须为同一张图中的ge.graph.Node对象。支持列表、元组和生成器等可迭代对象。 |

## 返回值说明

| 类型 | 说明 |
| --- | --- |
| FuseCheckResult | 所有检查均通过时返回`FuseCheckResult(ok=True, reason="")`；任一上述检查不满足时，返回`FuseCheckResult(ok=False, reason=<原因>)`。本接口不会自动输出原因，用户需要自行检查`reason`并根据需要输出。 |

当`nodes`中包含非ge.graph.Node对象时抛出`TypeError`；Node handle已失效时抛出`RuntimeError`。

不可融合的原因由用户自行输出，例如：

```python
from ge.passes import can_fuse

def fuse_nodes(nodes_before_fuse):
    # 执行can_fuse检查
    result = can_fuse(nodes_before_fuse)
    if not result.ok:
        print(f"[FuseNodes] can_fuse check failed: {result.reason}")
        return False
    return True
```

## 约束说明

- `nodes`中的节点需要属于同一张图。
- `nodes`不能为空，并且需要是待融合的连通节点集合。
- 业务原因导致不可融合时不抛出异常，应检查返回结果的`ok`和`reason`。

## 调用示例

`context`由GE在执行Pass时注入，是`FusionBasePass.run(graph, context)`的第二个参数，无需用户创建。

```python
from ge.passes import can_fuse

def run(self, graph, context):
    nodes_before = ...  # 从graph中选出的待融合节点
    result = can_fuse(nodes_before)
    if not result.ok:
        context.set_error_message(result.reason)
        return False
    ...
```
