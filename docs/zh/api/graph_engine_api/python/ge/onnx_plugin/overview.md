# 简介

`ge.onnx_plugin`提供编写ONNX自定义解析插件的Python接口。插件开发者声明ONNX原始算子到GE目标算子的映射并绑定解析回调，即可使用Python完成ONNX节点到GE Operator的转换，对应C++插件中的`REGISTER_CUSTOM_OP`注册和`ParseParamsFn`等回调实现。

插件文件放置在环境变量`ASCEND_CUSTOM_OPP_PATH`指定的路径下，由解析器在初始化阶段自动发现并注册。该环境变量支持以`:`分隔的多个路径；每个路径可以是单个`.py`文件，也可以是目录，目录模式下仅加载其中的直接`.py`文件和子包，名称以`_`开头的文件或目录会被跳过。同一个origin type（`domain::opset::算子类型`，例如`ai.onnx::11::Elu`）只能由一个插件提供；C++插件与Python插件冲突时保留C++插件，Python插件之间冲突时解析器初始化失败。

完整可运行的插件样例参见[ONNX Plugin样例](../../../../../../../examples/onnx_plugin/README.md)。

模块成员如下：

- [`onnx_plugin`](onnx_plugin.md)：创建插件descriptor。
- [`OnnxPlugin`](OnnxPlugin/overview.md)：插件descriptor，用于绑定解析回调。
- [`OnnxNode`](OnnxNode/overview.md)：ONNX节点的只读值对象，由GE作为回调实参传入。

以下示例注册Elu插件，读取ONNX节点的`alpha`属性并写入目标算子：

```python
from ge.graph import Operator
from ge.onnx_plugin import OnnxNode, onnx_plugin


elu = onnx_plugin(
    source="Elu",
    domain="ai.onnx",
    opsets=range(8, 19),
    target="Elu",
)


@elu.parse_node
def parse_elu(node: OnnxNode, target: Operator) -> None:
    target.set_attr("alpha", node.attrs.get("alpha", 1.0))
```
