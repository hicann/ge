# onnx\_plugin

## 产品支持情况

全量芯片支持。

## 功能说明

创建ONNX Plugin的descriptor对象，用于声明ONNX原始算子到GE目标算子的映射，并通过descriptor的[`parse_node`](OnnxPlugin/parse_node.md)、[`parse_operator`](OnnxPlugin/parse_operator.md)、[`decompose`](OnnxPlugin/decompose.md)方法绑定解析回调。

## 函数原型

```python
onnx_plugin(*, source: str, domain: str, opsets: Collection[int], target: str) -> OnnxPlugin
```

## 参数说明

| 参数名 | 输入/输出 | 描述 |
| :--- | :--- | :--- |
| source | 输入 | ONNX原始算子类型，例如`Elu`。类型为非空字符串，不允许包含`:`（`:`是origin type的字段分隔符，origin type按`<domain>::<opset>::<source>`格式拼接）。 |
| domain | 输入 | ONNX domain，标准ONNX算子使用`ai.onnx`。插件注册时必须显式声明，类型为非空字符串，不允许包含`:`（原因同`source`）。 |
| opsets | 输入 | 插件支持的ONNX opset版本。支持`list`、`tuple`、`set`、`range`等类型，`str`、`bytes`不支持。每个元素代表一个opset版本：`(1, 15)`即版本1和15，`range(1, 15)`即版本1至14（前闭后开）。元素必须为正整数，注册时自动去重并升序排列。 |
| target | 输入 | GE目标算子类型。类型为非空字符串，对应的算子原型必须已经安装并注册。 |

## 返回值说明

返回`OnnxPlugin`descriptor对象。

## 约束说明

- 参数校验在调用时立即执行。
- `source`、`domain`或`target`不是非空字符串时，抛出`TypeError`。
- `source`或`domain`包含`:`时，抛出`TypeError`（`:`是origin type的字段分隔符，出现会导致origin type无法按`<domain>::<opset>::<source>`切分）。
- `opsets`类型不受支持（如`int`、生成器），或为`str`、`bytes`时，抛出`TypeError`。
- `opsets`包含非整数元素（含`bool`）时，抛出`TypeError`。
- `opsets`为空或包含非正整数元素时，抛出`ValueError`。
- descriptor会将`source`、`domain`和`opsets`展开为完整的origin type集合（`domain::opset::source`，例如`ai.onnx::11::Elu`）。同一origin type只能由一个C++或Python插件提供。
- 这里的`domain`是插件作者注册时的声明，与ONNX模型文件中节点的domain字段无关：模型文件里节点的domain允许为空（ONNX标准规定为空即标准域`ai.onnx`），节点侧的显示规则参见[origin_type](OnnxNode/origin_type.md)。
- origin type冲突在解析器初始化阶段报告：Python插件与C++插件冲突时保留C++插件、拒绝Python插件；多个Python插件之间冲突时解析器初始化失败。

## 调用示例

```python
from ge.onnx_plugin import onnx_plugin

elu = onnx_plugin(
    source="Elu",
    domain="ai.onnx",
    opsets=range(8, 19),
    target="Elu",
)
```
