# infer\_shape

## 产品支持情况

全量芯片支持。

## 功能说明

用于在自定义融合Pass构造替换图后，对该[Graph](../graph/Graph/overview.md)执行[Shape](../graph/Shape/overview.md)、[DataType](../DataType.md)和[Format](../Format.md)推导。接口先根据`source`提供的原图边界信息，更新替换图输入Data节点的[TensorDesc](../graph/TensorDesc/overview.md)，再对替换图执行全图推导，原地更新图中算子的输出TensorDesc。`source`支持[MatchResult](MatchResult/overview.md)、[Node](../graph/Node/overview.md)和[SubgraphBoundary](SubgraphBoundary/overview.md)。

## 函数原型

```python
infer_shape(replacement: Graph, source: MatchResult | Node | SubgraphBoundary) -> None
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| replacement | 输入 | 待推导的替换图。接口会原地更新输入Data节点的TensorDesc，并通过全图推导更新图中算子的输出TensorDesc。 |
| source | 输入 | 原图边界信息的来源。[PatternFusionPass](PatternFusionPass/overview.md)场景传入MatchResult，[DecomposePass](DecomposePass/overview.md)场景传入原图中的Node；直接使用子图边界改图时，传入已构造的SubgraphBoundary。 |

## 返回值说明

成功时返回`None`。

- `TypeError`：`replacement`或`source`类型错误。
- `RuntimeError`：底层对象句柄失效、`replacement`为空图或推导失败。推导失败时，异常信息包含replacement graph名称、source类型，以及可获取时的source名称。

## 约束说明

- Pass注册在[PassStage](PassStage.md)的`BEFORE_INFER_SHAPE`阶段时，无需调用本接口。替换图接入原图后，其中的算子会参加GE编译流程中的全图`InferShapePass`。
- Pass注册在`AFTER_INFER_SHAPE`及之后阶段时，如果替换图引入了新算子，且其输出TensorDesc需要根据原图边界输入进行推导，则在返回替换图前调用本接口。
- 如果替换图未引入需要推导的算子，或者构图时已经完整、正确地设置了输入Data节点和各算子的TensorDesc，则无需调用本接口；调用方必须保证替换图的TensorDesc正确。
- 本接口不读取或校验当前PassStage。在任意阶段调用时都会立即执行推导，不会因为处于`BEFORE_INFER_SHAPE`阶段而报错或跳过。推导成功时返回`None`，推导失败时抛出`RuntimeError`。在`BEFORE_INFER_SHAPE`阶段显式调用本接口后，如果替换图成功接入原图，其中的算子还会参加后续GE编译流程中的全图`InferShapePass`。异常未被捕获时，当前replacement回调终止。
- `replacement`必须为Graph类型且不能为空图。
- `source`必须为MatchResult、Node或SubgraphBoundary类型。
- `replacement`和`source`的底层对象句柄必须有效。
- 替换图中的Data节点必须具有`index`属性，且`index`必须能在`source`提供的边界输入中找到对应项。
- `source`为MatchResult时，仅可在当前融合Pass的回调执行期间使用。回调结束后，MatchResult失效。

## 调用示例

```python
from ge.es import GraphBuilder
from ge.passes import infer_shape


# PatternFusionPass的replacement回调
def replacement(self, match_result):
    builder = GraphBuilder()
    replacement_input = builder.create_input(0)
    replacement = builder.build_and_reset([replacement_input])
    infer_shape(replacement, match_result)
    return replacement
```
